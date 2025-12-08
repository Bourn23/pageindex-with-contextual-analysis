import asyncio
import time
import logging

class Analyzer:
    def extract_keywords(self, node):
        return ["keyword"]

class Integrator:
    def __init__(self):
        self.semantic_analyzer = Analyzer()
        self.logger = logging.getLogger("Integrator")

    async def _process_nodes_recursive(self):
        # Mimic the structure
        start_time = time.time()
        print(f"Start time: {start_time}")
        
        # Hypothetical assignment that might cause issues?
        # time = 1  # If I uncomment this, it should fail line 13
        
        async def extract_keywords_async(node):
            try:
                st = time.time()
                print(f"Inner time: {st}")
            except Exception as e:
                print(f"Error: {e}")

        await extract_keywords_async({})

    async def run(self):
        await self._process_nodes_recursive()

async def main():
    i = Integrator()
    await i.run()

if __name__ == "__main__":
    asyncio.run(main())
