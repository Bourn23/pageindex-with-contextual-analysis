"""
Test script to verify LLM client works with both OpenAI and Gemini
"""
import os
import asyncio
from pageindex.llm_client import LLMClient, get_llm_client, set_llm_provider

def test_provider_detection():
    """Test automatic provider detection"""
    print("Testing provider detection...")
    print("=" * 60)
    
    # Check which keys are available
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY"))
    
    print(f"GEMINI_API_KEY present: {has_gemini}")
    print(f"OPENAI API key present: {has_openai}")
    
    if not has_gemini and not has_openai:
        print("\n⚠️  No API keys found!")
        print("Please add one of the following to your .env file:")
        print("  - GEMINI_API_KEY=your_key")
        print("  - CHATGPT_API_KEY=your_key")
        return False
    
    # Test auto-detection
    try:
        client = get_llm_client()
        print(f"\n✓ Auto-detected provider: {client.provider}")
        print(f"✓ Client initialized successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error initializing client: {e}")
        return False

def test_sync_completion():
    """Test synchronous completion"""
    print("\n\nTesting synchronous completion...")
    print("=" * 60)
    
    try:
        client = get_llm_client()
        
        # Use appropriate model based on provider
        if client.provider == 'gemini':
            model = 'gemini-1.5-flash'
        else:
            model = 'gpt-3.5-turbo'
        
        print(f"Using model: {model}")
        print("Sending test prompt...")
        
        response = client.chat_completion(
            model=model,
            prompt="Say 'Hello from PageIndex!' and nothing else."
        )
        
        print(f"\n✓ Response received: {response}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

async def test_async_completion():
    """Test asynchronous completion"""
    print("\n\nTesting asynchronous completion...")
    print("=" * 60)
    
    try:
        client = get_llm_client()
        
        # Use appropriate model based on provider
        if client.provider == 'gemini':
            model = 'gemini-1.5-flash'
        else:
            model = 'gpt-3.5-turbo'
        
        print(f"Using model: {model}")
        print("Sending async test prompt...")
        
        response = await client.chat_completion_async(
            model=model,
            prompt="Say 'Async works!' and nothing else."
        )
        
        print(f"\n✓ Response received: {response}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def test_finish_reason():
    """Test completion with finish reason"""
    print("\n\nTesting completion with finish reason...")
    print("=" * 60)
    
    try:
        client = get_llm_client()
        
        # Use appropriate model based on provider
        if client.provider == 'gemini':
            model = 'gemini-1.5-flash'
        else:
            model = 'gpt-3.5-turbo'
        
        print(f"Using model: {model}")
        print("Sending test prompt...")
        
        response, finish_reason = client.chat_completion_with_finish_reason(
            model=model,
            prompt="Count from 1 to 5."
        )
        
        print(f"\n✓ Response: {response}")
        print(f"✓ Finish reason: {finish_reason}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PageIndex LLM Client Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Provider detection
    results.append(("Provider Detection", test_provider_detection()))
    
    if not results[0][1]:
        print("\n⚠️  Skipping remaining tests (no API key found)")
        return
    
    # Test 2: Sync completion
    results.append(("Sync Completion", test_sync_completion()))
    
    # Test 3: Async completion
    try:
        async_result = asyncio.run(test_async_completion())
        results.append(("Async Completion", async_result))
    except Exception as e:
        print(f"Async test failed: {e}")
        results.append(("Async Completion", False))
    
    # Test 4: Finish reason
    results.append(("Finish Reason", test_finish_reason()))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
