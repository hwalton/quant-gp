# QuantGP Deployment Guide

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd quant-gp
   ```

2. **Install dependencies**:
   ```bash
   make install    # Installs Go deps and Air
   pip install -r requirements.txt  # Python deps
   ```

3. **Generate analysis**:
   ```bash
   ./update-analysis.sh  # Runs Python pipeline and copies images
   ```

4. **Start development server**:
   ```bash
   make dev  # Starts hot-reloading server on :8080
   ```

## Production Deployment

### Option 1: Simple VPS Deployment

1. **Build the application**:
   ```bash
   make build
   ```

2. **Copy to server**:
   ```bash
   scp -r bin/ web/ user@server:/path/to/app/
   ```

3. **Run on server**:
   ```bash
   ./bin/quantgp  # Runs on :8080
   ```

### Option 2: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN go build -o quantgp ./cmd

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/quantgp .
COPY --from=builder /app/web ./web
CMD ["./quantgp"]
EXPOSE 8080
```

Build and run:
```bash
docker build -t quantgp .
docker run -p 8080:8080 quantgp
```

### Option 3: Cloud Deployment

**Heroku**:
1. Create `Procfile`: `web: ./bin/quantgp`
2. Set buildpack: `heroku buildpacks:set heroku/go`
3. Deploy: `git push heroku main`

**Railway/Render**:
1. Connect GitHub repository
2. Set build command: `go build -o bin/quantgp ./cmd`
3. Set start command: `./bin/quantgp`

## Environment Variables

- `PORT`: Server port (default: 8080)

## Automated Analysis Updates

For production, set up a cron job to update analysis:

```bash
# Run analysis monthly on the 1st at 9 AM
0 9 1 * * cd /path/to/app && ./update-analysis.sh
```

## Monitoring

- Logs: Application logs to stdout
- Health check: `GET /` should return 200
- Static files: `GET /static/images/gp_output.png` should return image

## Security Considerations

For production:
1. Use a reverse proxy (nginx/Apache)
2. Enable HTTPS
3. Set up proper error logging
4. Consider rate limiting for signup endpoint
5. Add CORS headers if needed
6. Use environment variables for sensitive config
