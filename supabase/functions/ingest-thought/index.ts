import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const OPENROUTER_API_KEY = Deno.env.get("OPENROUTER_API_KEY")!;
const SLACK_BOT_TOKEN = Deno.env.get("SLACK_BOT_TOKEN")!;
const SLACK_CAPTURE_CHANNEL = Deno.env.get("SLACK_CAPTURE_CHANNEL")!;

const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

async function getEmbedding(text: string): Promise<number[]> {
  const r = await fetch(`${OPENROUTER_BASE}/embeddings`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: "openai/text-embedding-3-small", input: text }),
  });
  const d = await r.json();
  return d.data[0].embedding;
}

async function extractMetadata(text: string): Promise<Record<string, unknown>> {
  const r = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: `Extract metadata from the user's captured thought. Return JSON with:
- "people": array of people mentioned (empty if none)
- "action_items": array of implied to-dos (empty if none)
- "dates_mentioned": array of dates YYYY-MM-DD (empty if none)
- "topics": array of 1-3 short topic tags (always at least one)
- "type": one of "observation", "task", "idea", "reference", "person_note"
Only extract what's explicitly there.` },
        { role: "user", content: text },
      ],
    }),
  });
  const d = await r.json();
  try { return JSON.parse(d.choices[0].message.content); }
  catch { return { topics: ["uncategorized"], type: "observation" }; }
}

async function replyInSlack(channel: string, threadTs: string, text: string): Promise<void> {
  await fetch("https://slack.com/api/chat.postMessage", {
    method: "POST",
    headers: { "Authorization": `Bearer ${SLACK_BOT_TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify({ channel, thread_ts: threadTs, text }),
  });
}

const MCP_ACCESS_KEY = Deno.env.get("MCP_ACCESS_KEY") || "";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, content-type, x-client-info, apikey",
};

Deno.serve(async (req: Request): Promise<Response> => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", { status: 200, headers: corsHeaders });
  }

  try {
    const body = await req.json();

    // Slack url_verification
    if (body.type === "url_verification") {
      return new Response(JSON.stringify({ challenge: body.challenge }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Direct capture (from voice app or other non-Slack sources)
    if (body.content && body.source && body.source !== "slack") {
      // Verify access key
      const authHeader = req.headers.get("authorization") || "";
      const token = authHeader.replace("Bearer ", "");
      if (!token || token !== MCP_ACCESS_KEY) {
        return new Response(JSON.stringify({ error: "Unauthorized" }), {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      const [embedding, metadata] = await Promise.all([
        getEmbedding(body.content),
        extractMetadata(body.content),
      ]);

      const { error } = await supabase.from("thoughts").insert({
        content: body.content,
        embedding,
        metadata: { ...metadata, source: body.source },
      });

      if (error) {
        console.error("Direct capture insert error:", error);
        return new Response(JSON.stringify({ error: error.message }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      const meta = metadata as Record<string, unknown>;
      return new Response(JSON.stringify({
        status: "captured",
        type: meta.type || "thought",
        topics: meta.topics || [],
      }), {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Slack event handling
    const event = body.event;
    if (!event || event.type !== "message" || event.subtype || event.bot_id
        || event.channel !== SLACK_CAPTURE_CHANNEL) {
      return new Response("ok", { status: 200, headers: corsHeaders });
    }
    const messageText: string = event.text;
    const channel: string = event.channel;
    const messageTs: string = event.ts;
    if (!messageText || messageText.trim() === "") return new Response("ok", { status: 200, headers: corsHeaders });

    const [embedding, metadata] = await Promise.all([
      getEmbedding(messageText),
      extractMetadata(messageText),
    ]);

    const { error } = await supabase.from("thoughts").insert({
      content: messageText,
      embedding,
      metadata: { ...metadata, source: "slack", slack_ts: messageTs },
    });

    if (error) {
      console.error("Supabase insert error:", error);
      await replyInSlack(channel, messageTs, `Failed to capture: ${error.message}`);
      return new Response("error", { status: 500, headers: corsHeaders });
    }

    const meta = metadata as Record<string, unknown>;
    let confirmation = `Captured as *${meta.type || "thought"}*`;
    if (Array.isArray(meta.topics) && meta.topics.length > 0)
      confirmation += ` - ${meta.topics.join(", ")}`;
    if (Array.isArray(meta.people) && meta.people.length > 0)
      confirmation += `\nPeople: ${meta.people.join(", ")}`;
    if (Array.isArray(meta.action_items) && meta.action_items.length > 0)
      confirmation += `\nAction items: ${meta.action_items.join("; ")}`;

    await replyInSlack(channel, messageTs, confirmation);
    return new Response("ok", { status: 200, headers: corsHeaders });
  } catch (err) {
    console.error("Function error:", err);
    return new Response("error", { status: 500, headers: corsHeaders });
  }
});
