# items.py
import scrapy

class SECFilingItem(scrapy.Item):
    cik = scrapy.Field()
    company_name = scrapy.Field()
    filing_type = scrapy.Field()
    filing_date = scrapy.Field()
    filing_url = scrapy.Field()
    items = scrapy.Field()