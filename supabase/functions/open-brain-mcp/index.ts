import "jsr:@supabase/functions-js/edge-runtime.d.ts";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPTransport } from "@hono/mcp";
import { Hono } from "hono";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const OPENROUTER_API_KEY = Deno.env.get("OPENROUTER_API_KEY")!;
const MCP_ACCESS_KEY = Deno.env.get("MCP_ACCESS_KEY")!;

const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

async function getEmbedding(text: string): Promise<number[]> {
  const r = await fetch(`${OPENROUTER_BASE}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/text-embedding-3-small",
      input: text,
    }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => "");
    throw new Error(`OpenRouter embeddings failed: ${r.status} ${msg}`);
  }
  const d = await r.json();
  return d.data[0].embedding;
}

async function extractMetadata(text: string): Promise<Record<string, unknown>> {
  const r = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `Extract metadata from the user's captured thought. Return JSON with:
- "people": array of people mentioned (empty if none)
- "action_items": array of implied to-dos (empty if none)
- "dates_mentioned": array of dates YYYY-MM-DD (empty if none)
- "topics": array of 1-3 short topic tags (always at least one)
- "type": one of "observation", "task", "idea", "reference", "person_note"
Only extract what's explicitly there.`,
        },
        { role: "user", content: text },
      ],
    }),
  });
  const d = await r.json();
  try {
    return JSON.parse(d.choices[0].message.content);
  } catch {
    return { topics: ["uncategorized"], type: "observation" };
  }
}

// --- MCP Server Setup ---

const server = new McpServer({
  name: "open-brain",
  version: "1.0.0",
});

// Tool 1: Semantic Search
server.registerTool(
  "search_thoughts",
  {
    title: "Search Thoughts",
    description:
      "Search captured thoughts by meaning. Use this when the user asks about a topic, person, or idea they've previously captured.",
    inputSchema: {
      query: z.string().describe("What to search for"),
      limit: z.number().optional().default(10),
      threshold: z.number().optional().default(0.5),
    },
  },
  async ({ query, limit, threshold }) => {
    try {
      const qEmb = await getEmbedding(query);
      const { data, error } = await supabase.rpc("match_thoughts", {
        query_embedding: qEmb,
        match_threshold: threshold,
        match_count: limit,
        filter: {},
      });

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Search error: ${error.message}` }],
          isError: true,
        };
      }

      if (!data || data.length === 0) {
        return {
          content: [{ type: "text" as const, text: `No thoughts found matching "${query}".` }],
        };
      }

      const results = data.map(
        (
          t: {
            content: string;
            metadata: Record<string, unknown>;
            similarity: number;
            created_at: string;
          },
          i: number
        ) => {
          const m = t.metadata || {};
          const parts = [
            `--- Result ${i + 1} (${(t.similarity * 100).toFixed(1)}% match) ---`,
            `Captured: ${new Date(t.created_at).toLocaleDateString()}`,
            `Type: ${m.type || "unknown"}`,
          ];
          if (Array.isArray(m.topics) && m.topics.length)
            parts.push(`Topics: ${(m.topics as string[]).join(", ")}`);
          if (Array.isArray(m.people) && m.people.length)
            parts.push(`People: ${(m.people as string[]).join(", ")}`);
          if (Array.isArray(m.action_items) && m.action_items.length)
            parts.push(`Actions: ${(m.action_items as string[]).join("; ")}`);
          parts.push(`\n${t.content}`);
          return parts.join("\n");
        }
      );

      return {
        content: [
          {
            type: "text" as const,
            text: `Found ${data.length} thought(s):\n\n${results.join("\n\n")}`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 2: List Recent
server.registerTool(
  "list_thoughts",
  {
    title: "List Recent Thoughts",
    description:
      "List recently captured thoughts with optional filters by type, topic, person, or time range.",
    inputSchema: {
      limit: z.number().optional().default(10),
      type: z.string().optional().describe("Filter by type: observation, task, idea, reference, person_note"),
      topic: z.string().optional().describe("Filter by topic tag"),
      person: z.string().optional().describe("Filter by person mentioned"),
      days: z.number().optional().describe("Only thoughts from the last N days"),
    },
  },
  async ({ limit, type, topic, person, days }) => {
    try {
      let q = supabase
        .from("thoughts")
        .select("content, metadata, created_at")
        .order("created_at", { ascending: false })
        .limit(limit);

      if (type) q = q.contains("metadata", { type });
      if (topic) q = q.contains("metadata", { topics: [topic] });
      if (person) q = q.contains("metadata", { people: [person] });
      if (days) {
        const since = new Date();
        since.setDate(since.getDate() - days);
        q = q.gte("created_at", since.toISOString());
      }

      const { data, error } = await q;

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Error: ${error.message}` }],
          isError: true,
        };
      }

      if (!data || !data.length) {
        return { content: [{ type: "text" as const, text: "No thoughts found." }] };
      }

      const results = data.map(
        (
          t: { content: string; metadata: Record<string, unknown>; created_at: string },
          i: number
        ) => {
          const m = t.metadata || {};
          const tags = Array.isArray(m.topics) ? (m.topics as string[]).join(", ") : "";
          return `${i + 1}. [${new Date(t.created_at).toLocaleDateString()}] (${m.type || "??"}${tags ? " - " + tags : ""})\n   ${t.content}`;
        }
      );

      return {
        content: [
          {
            type: "text" as const,
            text: `${data.length} recent thought(s):\n\n${results.join("\n\n")}`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 3: Stats
server.registerTool(
  "thought_stats",
  {
    title: "Thought Statistics",
    description: "Get a summary of all captured thoughts: totals, types, top topics, and people.",
    inputSchema: {},
  },
  async () => {
    try {
      const { count } = await supabase
        .from("thoughts")
        .select("*", { count: "exact", head: true });

      const { data } = await supabase
        .from("thoughts")
        .select("metadata, created_at")
        .order("created_at", { ascending: false });

      const types: Record<string, number> = {};
      const topics: Record<string, number> = {};
      const people: Record<string, number> = {};

      for (const r of data || []) {
        const m = (r.metadata || {}) as Record<string, unknown>;
        if (m.type) types[m.type as string] = (types[m.type as string] || 0) + 1;
        if (Array.isArray(m.topics))
          for (const t of m.topics) topics[t as string] = (topics[t as string] || 0) + 1;
        if (Array.isArray(m.people))
          for (const p of m.people) people[p as string] = (people[p as string] || 0) + 1;
      }

      const sort = (o: Record<string, number>): [string, number][] =>
        Object.entries(o)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10);

      const lines: string[] = [
        `Total thoughts: ${count}`,
        `Date range: ${
          data?.length
            ? new Date(data[data.length - 1].created_at).toLocaleDateString() +
              " → " +
              new Date(data[0].created_at).toLocaleDateString()
            : "N/A"
        }`,
        "",
        "Types:",
        ...sort(types).map(([k, v]) => `  ${k}: ${v}`),
      ];

      if (Object.keys(topics).length) {
        lines.push("", "Top topics:");
        for (const [k, v] of sort(topics)) lines.push(`  ${k}: ${v}`);
      }

      if (Object.keys(people).length) {
        lines.push("", "People mentioned:");
        for (const [k, v] of sort(people)) lines.push(`  ${k}: ${v}`);
      }

      return { content: [{ type: "text" as const, text: lines.join("\n") }] };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 4: Capture Thought
server.registerTool(
  "capture_thought",
  {
    title: "Capture Thought",
    description:
      "Save a new thought to the Open Brain. Generates an embedding and extracts metadata automatically. Use this when the user wants to save something to their brain directly from any AI client — notes, insights, decisions, or migrated content from other systems.",
    inputSchema: {
      content: z.string().describe("The thought to capture — a clear, standalone statement that will make sense when retrieved later by any AI"),
    },
  },
  async ({ content }) => {
    try {
      const [embedding, metadata] = await Promise.all([
        getEmbedding(content),
        extractMetadata(content),
      ]);

      const { error } = await supabase.from("thoughts").insert({
        content,
        embedding,
        metadata: { ...metadata, source: "mcp" },
      });

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Failed to capture: ${error.message}` }],
          isError: true,
        };
      }

      const meta = metadata as Record<string, unknown>;
      let confirmation = `Captured as ${meta.type || "thought"}`;
      if (Array.isArray(meta.topics) && meta.topics.length)
        confirmation += ` — ${(meta.topics as string[]).join(", ")}`;
      if (Array.isArray(meta.people) && meta.people.length)
        confirmation += ` | People: ${(meta.people as string[]).join(", ")}`;
      if (Array.isArray(meta.action_items) && meta.action_items.length)
        confirmation += ` | Actions: ${(meta.action_items as string[]).join("; ")}`;

      return {
        content: [{ type: "text" as const, text: confirmation }],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// --- Ambient PA: Task Management Tools ---

// Tool 5: Create Task
server.registerTool(
  "create_task",
  {
    title: "Create Task",
    description:
      "Create a new task in the Ambient PA system. Use this when the user mentions something actionable, makes a commitment, or during brain dumps when implied to-dos are identified.",
    inputSchema: {
      title: z.string().describe("Short, actionable task title"),
      notes: z.string().optional().describe("Additional context or details"),
      entity: z.string().optional().describe("Entity: Oleinza, Circular, D Oehlmann, Natvia, WRS, Argus, TPM, TechVina, Personal"),
      project: z.string().optional().describe("Project grouping"),
      urgency: z.number().min(1).max(3).optional().default(2).describe("1=high/urgent, 2=medium, 3=low"),
      importance: z.number().min(1).max(3).optional().default(2).describe("1=high/important, 2=medium, 3=low"),
      due_date: z.string().optional().describe("Due date YYYY-MM-DD"),
      waiting_on: z.string().optional().describe("Who or what this task is blocked on"),
      source: z.string().optional().default("mcp").describe("Origin: mcp, slack, gmail, voice"),
      urgency_override: z.boolean().optional().default(false).describe("Lock urgency from auto-reclassification"),
      importance_override: z.boolean().optional().default(false).describe("Lock importance from auto-reclassification"),
    },
  },
  async ({ title, notes, entity, project, urgency, importance, due_date, waiting_on, source, urgency_override, importance_override }) => {
    try {
      const insert: Record<string, unknown> = {
        title,
        urgency,
        importance,
        urgency_override,
        importance_override,
        source,
        status: waiting_on ? "waiting" : "open",
      };
      if (notes) insert.notes = notes;
      if (entity) insert.entity = entity;
      if (project) insert.project = project;
      if (due_date) insert.due_date = due_date;
      if (waiting_on) insert.waiting_on = waiting_on;

      const { data, error } = await supabase
        .from("tasks")
        .insert(insert)
        .select("id, title, status")
        .single();

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Failed to create task: ${error.message}` }],
          isError: true,
        };
      }

      return {
        content: [{ type: "text" as const, text: `Task created: "${data.title}" [${data.status}] | ID: ${data.id}` }],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 6: List Tasks
server.registerTool(
  "list_tasks",
  {
    title: "List Tasks",
    description:
      "List tasks with optional filters. Returns active tasks by default (excludes done/deferred). Includes computed staleness_days for each task.",
    inputSchema: {
      status: z.string().optional().describe("Filter: open, in_progress, waiting, done, deferred"),
      entity: z.string().optional().describe("Filter by entity"),
      urgency: z.number().min(1).max(3).optional().describe("Filter by urgency: 1, 2, or 3"),
      importance: z.number().min(1).max(3).optional().describe("Filter by importance: 1, 2, or 3"),
      limit: z.number().optional().default(20),
    },
  },
  async ({ status, entity, urgency, importance, limit }) => {
    try {
      let q = supabase
        .from("tasks")
        .select("id, title, notes, entity, project, status, urgency, importance, urgency_override, importance_override, due_date, snoozed_until, snooze_count, max_snooze, waiting_on, source, created_at, last_surfaced_at")
        .limit(limit);

      if (status) {
        q = q.eq("status", status);
      } else {
        q = q.not("status", "in", '("done","deferred")');
      }
      if (entity) q = q.ilike("entity", `%${entity}%`);
      if (urgency) q = q.eq("urgency", urgency);
      if (importance) q = q.eq("importance", importance);

      const { data, error } = await q;

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Error: ${error.message}` }],
          isError: true,
        };
      }

      if (!data || !data.length) {
        return { content: [{ type: "text" as const, text: "No tasks found." }] };
      }

      const now = Date.now();
      const rows = data
        .map((t: Record<string, unknown>) => {
          const staleDays = Math.floor(
            (now - new Date((t.last_surfaced_at || t.created_at) as string).getTime()) / 86400000
          );
          return { ...t, staleness_days: staleDays };
        })
        .sort((a: Record<string, unknown>, b: Record<string, unknown>) =>
          (b.staleness_days as number) - (a.staleness_days as number)
        );

      const results = rows.map((t: Record<string, unknown>, i: number) => {
        const parts = [
          `${i + 1}. [${(t.status as string).toUpperCase()}] ${t.title} (U${t.urgency}/I${t.importance}, ${t.staleness_days}d stale)`,
        ];
        if (t.entity) parts.push(`   Entity: ${t.entity}`);
        if (t.project) parts.push(`   Project: ${t.project}`);
        if (t.due_date) parts.push(`   Due: ${t.due_date}`);
        if (t.waiting_on) parts.push(`   Waiting on: ${t.waiting_on}`);
        if (t.notes) parts.push(`   Notes: ${t.notes}`);
        if (t.snoozed_until) parts.push(`   Snoozed until: ${t.snoozed_until} (${t.snooze_count}/${t.max_snooze} snoozes)`);
        parts.push(`   ID: ${t.id}`);
        return parts.join("\n");
      });

      return {
        content: [
          {
            type: "text" as const,
            text: `${rows.length} task(s):\n\n${results.join("\n\n")}`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 7: Update Task
server.registerTool(
  "update_task",
  {
    title: "Update Task",
    description:
      "Update fields on an existing task. Use this to change status, reassign urgency/importance, add notes, set due dates, or record who we are waiting on.",
    inputSchema: {
      id: z.string().describe("Task UUID"),
      title: z.string().optional(),
      notes: z.string().optional(),
      entity: z.string().optional(),
      project: z.string().optional(),
      status: z.string().optional().describe("open, in_progress, waiting, done, deferred"),
      urgency: z.number().min(1).max(3).optional(),
      importance: z.number().min(1).max(3).optional(),
      urgency_override: z.boolean().optional(),
      importance_override: z.boolean().optional(),
      due_date: z.string().optional().describe("YYYY-MM-DD or empty string to clear"),
      snoozed_until: z.string().optional().describe("YYYY-MM-DD or empty string to clear"),
      waiting_on: z.string().optional().describe("Person/thing or empty string to clear"),
    },
  },
  async ({ id, title, notes, entity, project, status, urgency, importance, urgency_override, importance_override, due_date, snoozed_until, waiting_on }) => {
    try {
      const updates: Record<string, unknown> = {};
      if (title !== undefined) updates.title = title;
      if (notes !== undefined) updates.notes = notes;
      if (entity !== undefined) updates.entity = entity || null;
      if (project !== undefined) updates.project = project || null;
      if (status !== undefined) updates.status = status;
      if (urgency !== undefined) updates.urgency = urgency;
      if (importance !== undefined) updates.importance = importance;
      if (urgency_override !== undefined) updates.urgency_override = urgency_override;
      if (importance_override !== undefined) updates.importance_override = importance_override;
      if (due_date !== undefined) updates.due_date = due_date || null;
      if (snoozed_until !== undefined) updates.snoozed_until = snoozed_until || null;
      if (waiting_on !== undefined) updates.waiting_on = waiting_on || null;
      if (status === "done") updates.completed_at = new Date().toISOString();

      if (Object.keys(updates).length === 0) {
        return { content: [{ type: "text" as const, text: "No fields to update." }] };
      }

      const { data, error } = await supabase
        .from("tasks")
        .update(updates)
        .eq("id", id)
        .select("id, title, status, urgency, importance")
        .single();

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Failed to update: ${error.message}` }],
          isError: true,
        };
      }

      return {
        content: [
          {
            type: "text" as const,
            text: `Updated: "${data.title}" [${data.status}] U${data.urgency}/I${data.importance} | Changed: ${Object.keys(updates).join(", ")}`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 8: Complete Task
server.registerTool(
  "complete_task",
  {
    title: "Complete Task",
    description:
      "Mark a task as done. Sets status to done and records completion timestamp.",
    inputSchema: {
      id: z.string().describe("Task UUID"),
    },
  },
  async ({ id }) => {
    try {
      const { data, error } = await supabase
        .from("tasks")
        .update({
          status: "done",
          completed_at: new Date().toISOString(),
        })
        .eq("id", id)
        .select("id, title")
        .single();

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Failed to complete: ${error.message}` }],
          isError: true,
        };
      }

      return {
        content: [{ type: "text" as const, text: `Completed: "${data.title}"` }],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 9: Get Stale Tasks
server.registerTool(
  "get_stale_tasks",
  {
    title: "Get Stale Tasks",
    description:
      "Retrieve tasks needing attention based on staleness. Called at session start. Returns tasks where staleness >= threshold, or due_date <= today, or snooze_count >= max_snooze. Excludes snoozed tasks unless past max snooze. Tax/BAS tasks (Circular or D Oehlmann entities) due this month are always included. Updates last_surfaced_at on returned tasks.",
    inputSchema: {
      threshold_days: z.number().optional().default(4).describe("Minimum staleness days to include"),
    },
  },
  async ({ threshold_days }) => {
    try {
      const { data, error } = await supabase
        .from("tasks")
        .select("id, title, notes, entity, project, status, urgency, importance, urgency_override, importance_override, due_date, snoozed_until, snooze_count, max_snooze, waiting_on, created_at, last_surfaced_at")
        .not("status", "in", '("done","deferred")');

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Error: ${error.message}` }],
          isError: true,
        };
      }

      if (!data || !data.length) {
        return { content: [{ type: "text" as const, text: "No active tasks." }] };
      }

      const now = Date.now();
      const today = new Date().toISOString().split("T")[0];
      const currentYear = new Date().getFullYear();
      const currentMonth = new Date().getMonth();

      const TAX_ENTITIES = ["Circular", "D Oehlmann"];

      const stale = data
        .map((t: Record<string, unknown>) => {
          const staleDays = Math.floor(
            (now - new Date((t.last_surfaced_at || t.created_at) as string).getTime()) / 86400000
          );

          // Tax/BAS special case: entity is Circular or D Oehlmann
          // AND due_date falls within the current calendar month
          let taxDueThisMonth = false;
          const dueDate = t.due_date as string | null;
          const entity = t.entity as string | null;
          if (entity && dueDate && TAX_ENTITIES.includes(entity)) {
            const d = new Date(dueDate);
            if (d.getFullYear() === currentYear && d.getMonth() === currentMonth) {
              taxDueThisMonth = true;
            }
          }

          return { ...t, staleness_days: staleDays, tax_due_this_month: taxDueThisMonth };
        })
        .filter((t: Record<string, unknown>) => {
          // Tax/BAS: always include regardless of everything else
          if (t.tax_due_this_month) return true;

          const snoozedUntil = t.snoozed_until as string | null;
          const snoozeCount = t.snooze_count as number;
          const maxSnooze = t.max_snooze as number;
          const pastMaxSnooze = snoozeCount >= maxSnooze;
          const dueDate = t.due_date as string | null;
          const dueTodayOrPast = dueDate !== null && dueDate <= today;
          const staleDays = t.staleness_days as number;

          // Always include if due today/past or past max snooze
          if (dueTodayOrPast || pastMaxSnooze) return true;

          // Exclude if actively snoozed and under max snooze
          if (snoozedUntil && snoozedUntil > today && !pastMaxSnooze) return false;

          // Include if meets staleness threshold
          return staleDays >= threshold_days;
        })
        .sort((a: Record<string, unknown>, b: Record<string, unknown>) => {
          const scoreA = (a.staleness_days as number) * (a.urgency as number) * (a.importance as number);
          const scoreB = (b.staleness_days as number) * (b.urgency as number) * (b.importance as number);
          return scoreB - scoreA;
        })
        .slice(0, 10);

      if (!stale.length) {
        return { content: [{ type: "text" as const, text: "No stale tasks above threshold." }] };
      }

      // Update last_surfaced_at for all returned tasks
      const ids = stale.map((t: Record<string, unknown>) => t.id as string);
      await supabase
        .from("tasks")
        .update({ last_surfaced_at: new Date().toISOString() })
        .in("id", ids);

      const results = stale.map((t: Record<string, unknown>, i: number) => {
        const parts = [
          `${i + 1}. [${(t.status as string).toUpperCase()}]${t.tax_due_this_month ? " [TAX/BAS DUE]" : ""} ${t.title} (U${t.urgency}/I${t.importance}, ${t.staleness_days}d stale)`,
        ];
        if (t.entity) parts.push(`   Entity: ${t.entity}`);
        if (t.project) parts.push(`   Project: ${t.project}`);
        if (t.due_date) parts.push(`   Due: ${t.due_date}`);
        if (t.waiting_on) parts.push(`   Waiting on: ${t.waiting_on}`);
        if ((t.snooze_count as number) > 0)
          parts.push(`   Snoozed: ${t.snooze_count}/${t.max_snooze}${(t.snooze_count as number) >= (t.max_snooze as number) ? " — MAX REACHED" : ""}`);
        if (t.notes) parts.push(`   Notes: ${t.notes}`);
        parts.push(`   ID: ${t.id}`);
        return parts.join("\n");
      });

      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              tasks: stale.map((t: Record<string, unknown>) => ({
                id: t.id,
                title: t.title,
                entity: t.entity,
                project: t.project,
                status: t.status,
                urgency: t.urgency,
                importance: t.importance,
                urgency_override: t.urgency_override,
                importance_override: t.importance_override,
                due_date: t.due_date,
                snoozed_until: t.snoozed_until,
                snooze_count: t.snooze_count,
                max_snooze: t.max_snooze,
                waiting_on: t.waiting_on,
                last_surfaced_at: t.last_surfaced_at,
                staleness_days: t.staleness_days,
                tax_due_this_month: t.tax_due_this_month,
                created_at: t.created_at,
              })),
            }) + "\n\n" + results.join("\n\n"),
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 10: Snooze Task
server.registerTool(
  "snooze_task",
  {
    title: "Snooze Task",
    description:
      "Snooze a task for N days. If snooze_count has reached max_snooze, the snooze is rejected — the task requires a real decision (close, delegate, or set a due date).",
    inputSchema: {
      id: z.string().describe("Task UUID"),
      days: z.number().min(1).max(30).optional().default(3).describe("Days to snooze"),
    },
  },
  async ({ id, days }) => {
    try {
      const { data: task, error: fetchError } = await supabase
        .from("tasks")
        .select("id, title, snooze_count, max_snooze")
        .eq("id", id)
        .single();

      if (fetchError || !task) {
        return {
          content: [{ type: "text" as const, text: `Task not found: ${fetchError?.message || id}` }],
          isError: true,
        };
      }

      if (task.snooze_count >= task.max_snooze) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Task has reached maximum snoozes (${task.snooze_count}/${task.max_snooze}) — requires a real decision: close it, delegate it, or set a due date.`,
            },
          ],
          isError: true,
        };
      }

      const snoozeUntil = new Date();
      snoozeUntil.setDate(snoozeUntil.getDate() + days);
      const snoozeDate = snoozeUntil.toISOString().split("T")[0];
      const newCount = task.snooze_count + 1;
      const remaining = task.max_snooze - newCount;

      const { error: updateError } = await supabase
        .from("tasks")
        .update({
          snoozed_until: snoozeDate,
          snooze_count: newCount,
          last_surfaced_at: new Date().toISOString(),
        })
        .eq("id", id);

      if (updateError) {
        return {
          content: [{ type: "text" as const, text: `Failed to snooze: ${updateError.message}` }],
          isError: true,
        };
      }

      return {
        content: [
          {
            type: "text" as const,
            text: `Snoozed: "${task.title}" until ${snoozeDate} (${newCount}/${task.max_snooze} snoozes used, ${remaining} remaining)`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// --- Hono App with Auth Check ---

const app = new Hono();

app.all("*", async (c) => {
  // Accept access key via header OR URL query parameter
  const provided = c.req.header("x-brain-key") || new URL(c.req.url).searchParams.get("key");
  if (!provided || provided !== MCP_ACCESS_KEY) {
    return c.json({ error: "Invalid or missing access key" }, 401);
  }

  const transport = new StreamableHTTPTransport();
  await server.connect(transport);
  return transport.handleRequest(c);
});

Deno.serve(app.fetch);
