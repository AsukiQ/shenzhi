import json
import tempfile
import unittest
from pathlib import Path
from paper_search import PaperSearchIndex, SearchFilters
from rebuild_search_index import rebuild


class IndexRoundtripTest(unittest.TestCase):
    def test_metadata_and_funding_recall_survive_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); docs=root/'docs.jsonl'
            docs.write_text(json.dumps(dict(paper_id='p',title='Example',abstract='Research',
                authors=['Alice'],keywords=['testing'],subjects=['cognition'],
                funding=['UniqueGrantXYZ'],image_id='example.png'))+'\n')
            index=PaperSearchIndex(root/'source.db'); index.build(docs); index.close()
            self.assertEqual(rebuild(root/'source.db',root/'target.db'),1)
            fresh=PaperSearchIndex(root/'target.db')
            results,_=fresh.search('UniqueGrantXYZ',filters=SearchFilters(author=['Alice'],subject=['cognition']))
            self.assertEqual(len(results),1)
            self.assertEqual(results[0].funding,['UniqueGrantXYZ'])
            self.assertEqual(results[0].image_id,'example.png')
            fresh.close()
            with self.assertRaises(ValueError): rebuild(root/'source.db',root/'target.db')
