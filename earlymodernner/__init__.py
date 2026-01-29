"""
EarlyModernNER - Named Entity Recognition for Early Modern English Documents

A specialized NER tool for extracting entities from historical documents (1500-1800).

Entity types:
- TOPONYM: Place names (cities, ports, regions, countries)
- PERSON: Individual people (authors, merchants, officials)
- ORGANIZATION: Institutions (companies, guilds, courts)
- COMMODITY: Trade goods, foodstuffs, materials

Usage:
    # Command line
    python -m earlymodernner --input /path/to/docs --output results.jsonl

    # Python API
    from earlymodernner.pipeline import run_pipeline
    run_pipeline(input_path, output_path)
"""

from .version import VERSION

__version__ = VERSION
__all__ = ["VERSION", "__version__"]
