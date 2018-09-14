from numpy import array_equal
from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.measure import Measure
from torch_kalman.process.velocity import Velocity

from torch_kalman.tests import TestCaseTK

from scipy.linalg import block_diag


class TestDesign(TestCaseTK):
    def test_design_attrs(self):
        with self.assertRaises(ValueError) as cm:
            design = Design(measures=[Measure(id='same'), Measure(id='same')], processes=[])
        self.assertEqual(cm.exception.args[0], "Duplicate measure-ids: same.")

        with self.assertRaises(ValueError) as cm:
            design = Design(processes=[Velocity(id='same'), Velocity(id='same')], measures=[])
        self.assertEqual(cm.exception.args[0], "Duplicate process-ids: same.")

    @staticmethod
    def make_usable_design(dims=2):
        # create design:
        processes, measures = [], []
        for i in range(dims):
            process = Velocity(id=str(i))
            measure = Measure(id=str(i))
            measure.add_process(process, value=1.)
            processes.append(process)
            measures.append(measure)
        return Design(processes=processes, measures=measures)

    def test_design_f(self):
        # design
        design = self.make_usable_design()
        batch_design = design.for_batch(1)
        batch_design.lock()

        # F doesn't require grad:
        self.assertFalse(batch_design.F.requires_grad)

        # F is block diagonal of components:
        design_F = batch_design.F[0].data.numpy()
        manual_F = block_diag(*[process.F[0].data.numpy() for process in batch_design.processes.values()])
        self.assertTrue(array_equal(design_F, manual_F))

    def test_design_q(self):
        # design
        design = self.make_usable_design()
        batch_design = design.for_batch(1)
        batch_design.lock()

        # Q requires grad:
        self.assertTrue(batch_design.Q.requires_grad)

        # symmetric
        design_Q = batch_design.Q[0].data.numpy()
        self.assertTrue(array_equal(design_Q, design_Q.T), msg="Covariance is not symmetric.")

        # block diag
        manual_Q = block_diag(*[process.Q[0].data.numpy() for process in batch_design.processes.values()])
        self.assertTrue(array_equal(design_Q, manual_Q))

    def test_design_h(self):
        # design
        design = self.make_usable_design()
        batch_design = design.for_batch(1)
        batch_design.lock()

        design_H = batch_design.H
        state_mean = Tensor([[[1.], [-.5], [-.5], [0.]]])
        measured_state = design_H.bmm(state_mean)
        self.assertListEqual(list1=measured_state.tolist(), list2=[[[1.0], [-.5]]])

    def test_design_r(self):
        design = self.make_usable_design(3)
        batch_design = design.for_batch(1)
        batch_design.lock()

        cov = batch_design.R[0]
        self.assertTupleEqual(cov.size(), (3, 3))

        self.assertTrue(cov.requires_grad)

        self.check_covariance_chol(cov=cov,
                                   cholesky_log_diag=design.measure_cholesky_log_diag,
                                   cholesky_off_diag=design.measure_cholesky_off_diag)
