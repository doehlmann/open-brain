const CACHE_NAME = 'open-brain-voice-v1';
const ASSETS = ['/open-brain/', '/open-brain/index.html', '/open-brain/manifest.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  // Only cache app shell, never cache API calls
  if (e.request.url.includes('supabase.co')) return;

  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
