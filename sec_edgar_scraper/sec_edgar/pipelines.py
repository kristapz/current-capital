import json

class SecEdgarPipeline:
    def process_item(self, item, spider):
        return item

    def open_spider(self, spider):
        self.file = open('10k_filings.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item