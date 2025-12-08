#!/usr/bin/env python3
import asyncio
from pathlib import Path

async def test():
    print('Starting test...')
    md_file = 'tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md'
    
    print(f'File exists: {Path(md_file).exists()}')
    
    print('Reading file...')
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f'File size: {len(content)} chars')
    print(f'First 200 chars: {repr(content[:200])}')
    
    print('\nTesting markdown-it parser...')
    from markdown_it import MarkdownIt
    md = MarkdownIt()
    tokens = md.parse(content)
    print(f'Parsed {len(tokens)} tokens')
    print('Test complete')

asyncio.run(test())
