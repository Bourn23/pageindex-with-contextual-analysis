"""Check if config is being set correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.utils import ConfigLoader

# Test medium granularity config
config_loader = ConfigLoader()
opt = config_loader.load({
    'model': 'gemini-2.5-flash-lite',
    'granularity': 'medium',
    'enable_figure_detection': True,
    'enable_table_detection': True,
    'enable_semantic_subdivision': True,
    'if_add_node_id': 'yes',
    'if_add_node_summary': 'no',
    'if_add_doc_description': 'no'
})

print("Config check:")
print(f"  granularity: {opt.granularity}")
print(f"  enable_figure_detection: {opt.enable_figure_detection}")
print(f"  enable_table_detection: {opt.enable_table_detection}")
print(f"  enable_semantic_subdivision: {opt.enable_semantic_subdivision}")
print(f"  semantic_min_pages: {getattr(opt, 'semantic_min_pages', 'NOT SET')}")

print("\nAll config attributes:")
for attr in dir(opt):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(opt, attr)}")
