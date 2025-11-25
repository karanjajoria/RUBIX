# How to Get Your Gemini API Key

The demo requires a **free** Google Gemini API key. Here's how to get one:

## Step 1: Get Your API Key (2 minutes)

1. **Go to Google AI Studio:**
   https://aistudio.google.com/app/apikey

2. **Sign in** with your Google account

3. **Click "Create API Key"**

4. **Copy the API key** (it will look like: `AIzaSyA...`)

## Step 2: Add to Your .env File

1. Open the `.env` file in your project root

2. Replace this line:
   ```
   GEMINI_API_KEY=
   ```

   With your actual key:
   ```
   GEMINI_API_KEY=AIzaSyA_your_actual_key_here
   ```

3. **Save the file**

## Step 3: Run the Demo

```bash
python main.py --mode demo
```

## Important Notes

- ✅ **Free tier**: 60 requests per minute (more than enough for demo)
- ✅ **No credit card required**
- ✅ **Works immediately** after creation
- ⚠️ **Keep it secret**: Don't commit .env to git (already in .gitignore)

## Troubleshooting

**Error: "No API_KEY or ADC found"**
- Make sure you saved the .env file after adding the key
- Restart your terminal/command prompt
- Check there are no extra spaces around the key

**Error: "429 quota exceeded"**
- You're hitting rate limits (unlikely in demo)
- Wait 1 minute and try again
- Free tier allows 60 requests/minute

## What the API is Used For

The Gemini API is used by 4 agents:

1. **Vision Agent** - Multi-modal analysis (image + text)
2. **Forecasting Agent** - Summarizing displacement predictions
3. **Communication Agent** - Translating alerts to 4 languages
4. **Orchestrator** - Conflict resolution between agents

## Alternative: Run Without Gemini (Limited)

If you can't get the API key right now, you can:

1. Comment out Gemini calls (agents will use fallbacks)
2. System will still demonstrate multi-agent architecture
3. But outputs will be less intelligent

---

**Need help?** Check the official docs: https://ai.google.dev/gemini-api/docs/api-key
