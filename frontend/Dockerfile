# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Build arguments for environment variables
ARG VITE_API_URL=http://localhost:8000
ARG VITE_WS_URL=ws://localhost:8000

# Set environment variables for build
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL

# Copy package files first for dependency caching
COPY package*.json ./

# Install dependencies (cached unless package.json changes)
RUN npm ci --prefer-offline --no-audit --no-fund

# Copy source code AFTER deps (this is what changes most often)
COPY . .

# Build the application
RUN npm run build && echo "Build completed at $(date)"

# Production stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
