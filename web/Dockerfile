# Showcase site image, built on `web` by deploy/deploy.sh (see deploy/compose.yml).
# Node matches ai's version; node:sqlite needs >= 22.13.
ARG NODE_VERSION=22.23.2
FROM node:${NODE_VERSION}-slim

WORKDIR /app
ENV NODE_ENV=production

COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

# tsx reads tsconfig.json for the hono/jsx import source, so it ships with the sources.
COPY tsconfig.json ./
COPY src ./src
COPY public ./public

USER node
EXPOSE 3000
CMD ["node", "--import", "tsx", "src/server.ts"]
