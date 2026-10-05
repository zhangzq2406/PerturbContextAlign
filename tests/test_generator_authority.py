import hashlib,unittest
from pathlib import Path
class GeneratorAuthority(unittest.TestCase):
    def test_frozen_generator_sha(self):
        p=Path(__file__).resolve().parents[1]/'src/context/preprocess2_embedding_bgem3_p4_v4_context_instance.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'dceb6851609c839680157ed5d55ca0042d02c446cf92d8cf507eccfe7d4d8937')
