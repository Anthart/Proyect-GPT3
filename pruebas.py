from gpt3 import Gpt3
import unittest


class TestGPT3Proyect(unittest.TestCase):
    probs = [
        {
            'easy': 0.9427575484820901,
            'very': 0.024555334023266522,
            'Easy': 0.0011065233935420998,
            'very easy': 0.0006448358179143308
        },
        {
            'difficult': 0.9046915745232976,
            'neutral': 0.06653019692325582,
            'easy': 0.016178786017296717,
            'very': 0.010345982488383636
        },
        {
            'easy': 0.5,
            'difficult': 0.4,
            'very easy': 0.09
        },
        {
            'neutral': 0.5,
            'easy': 0.3,
            'difficult': 0.2
        }
    ]

    def test_strat3_1(self):
        resultado = Gpt3.strat_3("easy", self.probs[0])
        self.assertEqual(resultado, 0.26414940392499886)

    def test_strat3_2(self):
        resultado = Gpt3.strat_3("difficult", self.probs[1])
        self.assertEqual(resultado, 0.7468501393659625)

    # def test_strat3_3(self):
    #     resultados = Gpt3.strat_3("easy", self.probs[2])
    #     self.assertEqual(resultados, 0.45249)

    def test_strat_4(self):
        resultado = Gpt3.strat_3("neutral", self.probs[3])
        self.assertEqual(resultado, 0.475)


unittest.main()
