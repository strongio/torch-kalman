from torch_kalman.design import Design
from torch_kalman.measure import Measure

from torch_kalman.tests import TestCaseTK


class TestDesign(TestCaseTK):
    def test_design_attrs(self):
        with self.assertRaises(ValueError) as cm:
            design = Design(measures=[Measure(id='same'), Measure(id='same')], processes=[])
        self.assertEqual(cm.exception.args[0], "Duplicate measure-ids: same.")

    def test_design_f(self):
        # TODO
        pass

    def test_design_q(self):
        # TODO
        pass

    def test_design_h(self):
        # TODO
        pass

    def test_design_r(self):
        measures = []
        for i in range(3):
            measures.append(Measure(id=f'test{i}'))

        design = Design(measures=measures, processes=[])

        self.check_covariance_chol(cov=design.measure_covariance(),
                                   cholesky_log_diag=design.measure_cholesky_log_diag,
                                   cholesky_off_diag=design.measure_cholesky_off_diag)
