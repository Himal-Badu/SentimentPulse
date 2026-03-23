# Deployment Guide - SentimentPulse

## 🚀 Deployment Options

### Option 1: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or use the CLI
sentiment analyze "Your text here"
```

### Option 2: Docker

```bash
# Build the image
docker build -t sentimentpulse .

# Run container
docker run -p 8000:8000 sentimentpulse

# Or use docker-compose
docker-compose up -d
```

### Option 3: Cloud Platforms

#### Railway
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### Render
1. Connect your GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

#### Heroku
```bash
heroku create sentimentpulse
git push heroku master
```

## 📋 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `info` |

## 🧪 Testing Production Build

```bash
# Build Docker
docker build -t sentimentpulse:test .

# Run tests
docker run sentimentpulse:test pytest tests/

# Test API
docker run -p 8000:8000 sentimentpulse:test
curl http://localhost:8000/health
```

## 🔒 Production Considerations

1. **Use a virtual environment** for isolation
2. **Set proper CORS origins** in production
3. **Add rate limiting** for API endpoints
4. **Use HTTPS** in production
5. **Monitor logs** and set up alerts
6. **Scale horizontally** with multiple instances

## 📞 Support

- GitHub Issues: https://github.com/Himal-Badu/SentimentPulse/issues
- Documentation: https://github.com/Himal-Badu/SentimentPulse#readme
