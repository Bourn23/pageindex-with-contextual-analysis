#!/usr/bin/env python3
import asyncio

async def test():
    print('Async function started')
    await asyncio.sleep(0.1)
    print('Async function completed')

print('Before asyncio.run')
asyncio.run(test())
print('After asyncio.run')
