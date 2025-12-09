# ============================================================================
# Multi-stage Dockerfile for Real Estate AI Analysis Platform - Frontend
# ============================================================================
# Stage 1: Dependencies stage for installing npm packages
# ============================================================================

FROM node:20-alpine AS deps

# Set working directory
WORKDIR /app

# Install dependencies only when needed
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# ============================================================================
# Stage 2: Builder stage for building the Next.js application
# ============================================================================

FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy application source code
COPY . .

# Create public directory if it doesn't exist (Next.js may need it)
RUN mkdir -p public

# Set environment variables for build
# Next.js will use these during build time
ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

# Build the Next.js application
# Using standalone output for optimized production build
RUN npm run build

# ============================================================================
# Stage 3: Runner stage with Nginx for serving static files
# ============================================================================

FROM nginx:alpine AS runner

WORKDIR /app

# Copy built application from builder stage
# Next.js standalone output structure:
# - .next/standalone/ contains the Node.js server
# - .next/static/ contains static assets
# - public/ contains public files
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Install Node.js and curl for health checks
RUN apk add --no-cache nodejs curl

# Create nginx configuration for Next.js
RUN echo 'server { \
    listen 80; \
    server_name _; \
    root /app; \
    \
    # Gzip compression \
    gzip on; \
    gzip_vary on; \
    gzip_min_length 1024; \
    gzip_comp_level 6; \
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript application/json image/svg+xml; \
    \
    # Security headers \
    add_header X-Frame-Options "SAMEORIGIN" always; \
    add_header X-Content-Type-Options "nosniff" always; \
    add_header X-XSS-Protection "1; mode=block" always; \
    add_header Referrer-Policy "no-referrer-when-downgrade" always; \
    \
    # Cache static assets (immutable) \
    location /_next/static { \
        alias /app/.next/static; \
        expires 365d; \
        add_header Cache-Control "public, immutable"; \
        access_log off; \
    } \
    \
    # Serve public files \
    location /public { \
        alias /app/public; \
        expires 30d; \
        add_header Cache-Control "public"; \
    } \
    \
    # Proxy to Next.js standalone server \
    location / { \
        proxy_pass http://127.0.0.1:3000; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_cache_bypass $http_upgrade; \
        proxy_connect_timeout 60s; \
        proxy_send_timeout 60s; \
        proxy_read_timeout 60s; \
    } \
    \
    # Health check endpoint \
    location /health { \
        access_log off; \
        return 200 "healthy\n"; \
        add_header Content-Type text/plain; \
    } \
}' > /etc/nginx/conf.d/default.conf

# Create startup script to run both Node.js server and nginx
# The script waits for the Node.js server to be ready before starting nginx
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Verify server.js exists' >> /app/start.sh && \
    echo 'if [ ! -f /app/server.js ]; then' >> /app/start.sh && \
    echo '  echo "ERROR: server.js not found in /app directory"' >> /app/start.sh && \
    echo '  echo "Contents of /app:"' >> /app/start.sh && \
    echo '  ls -la /app' >> /app/start.sh && \
    echo '  exit 1' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Set environment variables for Next.js server' >> /app/start.sh && \
    echo 'export PORT=3000' >> /app/start.sh && \
    echo 'export HOSTNAME="0.0.0.0"' >> /app/start.sh && \
    echo 'export HOST="0.0.0.0"' >> /app/start.sh && \
    echo 'export NODE_ENV="production"' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start Next.js server in background' >> /app/start.sh && \
    echo 'cd /app' >> /app/start.sh && \
    echo 'echo "Starting Next.js server on port 3000..."' >> /app/start.sh && \
    echo 'node server.js > /tmp/node-server.log 2>&1 &' >> /app/start.sh && \
    echo 'NODE_PID=$!' >> /app/start.sh && \
    echo 'echo "Node.js server started with PID: $NODE_PID"' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Wait for server to be ready (max 30 seconds)' >> /app/start.sh && \
    echo 'echo "Waiting for Next.js server to be ready..."' >> /app/start.sh && \
    echo 'for i in $(seq 1 30); do' >> /app/start.sh && \
    echo '  if curl -f http://127.0.0.1:3000 > /dev/null 2>&1; then' >> /app/start.sh && \
    echo '    echo "Next.js server is ready!"' >> /app/start.sh && \
    echo '    break' >> /app/start.sh && \
    echo '  fi' >> /app/start.sh && \
    echo '  # Check if process is still running' >> /app/start.sh && \
    echo '  if ! kill -0 $NODE_PID 2>/dev/null; then' >> /app/start.sh && \
    echo '    echo "ERROR: Node.js server process died. Logs:"' >> /app/start.sh && \
    echo '    cat /tmp/node-server.log' >> /app/start.sh && \
    echo '    exit 1' >> /app/start.sh && \
    echo '  fi' >> /app/start.sh && \
    echo '  if [ $i -eq 30 ]; then' >> /app/start.sh && \
    echo '    echo "ERROR: Next.js server failed to start within 30 seconds"' >> /app/start.sh && \
    echo '    echo "Server logs:"' >> /app/start.sh && \
    echo '    cat /tmp/node-server.log' >> /app/start.sh && \
    echo '    kill $NODE_PID 2>/dev/null || true' >> /app/start.sh && \
    echo '    exit 1' >> /app/start.sh && \
    echo '  fi' >> /app/start.sh && \
    echo '  sleep 1' >> /app/start.sh && \
    echo 'done' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Function to handle shutdown' >> /app/start.sh && \
    echo 'cleanup() {' >> /app/start.sh && \
    echo '  echo "Shutting down..."' >> /app/start.sh && \
    echo '  kill $NODE_PID 2>/dev/null || true' >> /app/start.sh && \
    echo '  exit 0' >> /app/start.sh && \
    echo '}' >> /app/start.sh && \
    echo 'trap cleanup SIGTERM SIGINT' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start nginx in foreground' >> /app/start.sh && \
    echo 'echo "Starting nginx..."' >> /app/start.sh && \
    echo 'exec nginx -g "daemon off;"' >> /app/start.sh && \
    chmod +x /app/start.sh

# Create non-root user (nginx already runs as nginx user)
RUN chown -R nginx:nginx /app && \
    chmod -R 755 /app

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

# Start both Next.js server and nginx
CMD ["/app/start.sh"]
