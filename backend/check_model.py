import google.generativeai as genai
import os

# 🛑 PASTE YOUR API KEY HERE
MY_API_KEY = "AIzaSyAw0e5tgWvqMK_2-IHZZaBrTB399jTCSrk"

genai.configure(api_key=MY_API_KEY)

print("🔍 Checking for available Gemini models...")

try:
    found_any = False
    for m in genai.list_models():
        # We only care about models that can write text (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"   ✅ AVAILABLE: {m.name}")
            found_any = True
    
    if not found_any:
        print("❌ No text generation models found. Check your API Key permissions.")

except Exception as e:
    print(f"🔥 Error connecting to Google: {e}")