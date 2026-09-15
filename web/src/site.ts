/** Deployment settings, read on every call so tests can change them. */
export function siteConfig() {
  return {
    /** Public address of the site, e.g. https://localinference.alexwoodka.com; used for og:url. */
    publicOrigin: (process.env.PUBLIC_ORIGIN ?? '').replace(/\/+$/, ''),
    /** Open WebUI behind Cloudflare Access, e.g. https://chat.alexwoodka.com. */
    chatUrl: process.env.CHAT_URL || undefined,
  };
}
