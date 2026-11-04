import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_vlm.data.raw_labels import match_raw_category
class RawSemanticTests(unittest.TestCase):
 def setUp(self): self.patterns={'ic':3,'led':5,'capacitor':1,'resistor':2}; self.ignore=['text','pads','pins','test','jumper','unknown']
 def test_text_led_designator_rejected(self): self.assertEqual(match_raw_category('text "LED4/P1.21"',self.patterns,self.ignore)[0],0)
 def test_text_incidental_substrings_rejected(self):
  for s in ['text "ASSEMBLED IN USA"','text "ELECTRONICS PLATFORM"','text COMMUNICATION','"component text" "IC LED"']:
   self.assertEqual(match_raw_category(s,self.patterns,self.ignore)[0],0)
 def test_real_components_with_unknown_values_retained(self):
  for s,c in [('ic UNKNOWN',3),('led LED4',5),('capacitor unknown',1),('"resistor network" unknown',2)]:
   self.assertEqual(match_raw_category(s,self.patterns,self.ignore)[0],c)
 def test_value_cannot_override_unknown_class(self): self.assertEqual(match_raw_category('unknown LED4',self.patterns,self.ignore)[0],0)
 def test_jumper_compound_follows_ignore_policy(self): self.assertEqual(match_raw_category('"capacitor jumper" x',self.patterns,self.ignore)[0],0)
 def test_malformed_label_rejected(self): self.assertEqual(match_raw_category('"text LED4',self.patterns,self.ignore)[0],0)
