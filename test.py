try:
    import pyscf
    with_pyscf = True
except ImportError:
    with_pyscf = False

from cqcpy.tests.test_cc_ampl import *
from cqcpy.tests.test_cc_rdm import *
from cqcpy.tests.test_ci_utils import *
from cqcpy.tests.test_ft_utils import *
if with_pyscf:
    from cqcpy.tests.test_integrals import *
from cqcpy.tests.test_lambda_equations import *
from cqcpy.tests.test_spin_utils import *
from cqcpy.tests.test_test import *

if __name__ == '__main__':
    unittest.main()
