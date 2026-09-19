# NEXT_PUBLIC_API_BASE is inlined into the client bundle at build time, so it
# must be passed as a build arg, not a run-time -e flag:
#   docker build --build-arg NEXT_PUBLIC_API_BASE=http://localhost:8000 -t insurance-claims-frontend .
#   docker run -p 3000:3000 insurance-claims-frontend
FROM node:20-slim AS build
WORKDIR /app
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app ./
EXPOSE 3000
CMD ["npm", "start"]
