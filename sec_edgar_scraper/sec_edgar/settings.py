BOT_NAME = 'sec_edgar'

SPIDER_MODULES = ['sec_edgar.spiders']
NEWSPIDER_MODULE = 'sec_edgar.spiders'

ROBOTSTXT_OBEY = False

CONCURRENT_REQUESTS = 1

DOWNLOAD_DELAY = 10

LOG_LEVEL = 'INFO'

COOKIES_ENABLED = False

USER_AGENT = 'Your Name yourname@example.com'

ITEM_PIPELINES = {
   'sec_edgar.pipelines.SecEdgarPipeline': 300,
}

FEED_EXPORT_ENCODING = 'utf-8'