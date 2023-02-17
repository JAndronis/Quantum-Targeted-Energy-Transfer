import unittest
from src.tet.HamiltonianLoss import Loss
import src.tet.constants as constants

CONST = constants.system_constants
ACCEPTOR = constants.acceptor


class TestLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._loss = Loss(CONST)

    def test_init(self):
        self.assertEqual(self._loss.targetState, CONST['sites'] - 1)
        self.assertEqual(self._loss.max_N_np, CONST['max_N'])
        self.assertEqual(self._loss.max_t.numpy(), CONST['max_t'])
        self.assertEqual(self._loss.coupling_lambda.numpy(), CONST['coupling'])
        self.assertListEqual(list(self._loss.omegas.numpy()), CONST['omegas'])

    def test_evolve(self):
        xa = (CONST['omegas'][0] - CONST['omegas'][1]) / CONST['max_N']
        xd = (CONST['omegas'][1] - CONST['omegas'][0]) / CONST['max_N']
        chis = [xd, xa]
        self.assertLessEqual(self._loss(chis, site=ACCEPTOR, single_value=True).numpy(), 0.1)

if __name__=="__main__":
    unittest.main()