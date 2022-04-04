import warnings
warnings.filterwarnings('ignore')

from fastai.vision import *
from sklearn import metrics
import argparse
import pickle
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

test_id = ['test_neg_1', 'test_neg_2', 'test_neg_3', 'test_neg_4', 'test_neg_5', 'test_neg_6', 'test_neg_7', 'test_neg_8', 'test_neg_9', 'test_neg_10', 'test_neg_11', 'test_neg_12', 'test_neg_13', 'test_neg_14', 'test_neg_15', 'test_neg_16', 'test_neg_17', 'test_neg_18', 'test_neg_19', 'test_neg_20', 'test_neg_21', 'test_neg_22', 'test_neg_23', 'test_neg_24', 'test_neg_25', 'test_neg_26', 'test_neg_27', 'test_neg_28', 'test_neg_29', 'test_neg_30', 'test_neg_31', 'test_neg_32', 'test_neg_33', 'test_neg_34', 'test_neg_35', 'test_neg_36', 'test_neg_37', 'test_neg_38', 'test_neg_39', 'test_neg_40', 'test_neg_41', 'test_neg_42', 'test_neg_43', 'test_neg_44', 'test_neg_45', 'test_neg_46', 'test_neg_47', 'test_neg_48', 'test_neg_49', 'test_neg_50', 'test_neg_51', 'test_neg_52', 'test_neg_53', 'test_neg_54', 'test_neg_55', 'test_neg_56', 'test_neg_57', 'test_neg_58', 'test_neg_59', 'test_pos_1', 'test_pos_2', 'test_pos_3', 'test_pos_4', 'test_pos_5', 'test_pos_6', 'test_pos_7', 'test_pos_8', 'test_pos_9', 'test_pos_10', 'test_pos_11', 'test_pos_12', 'test_pos_13', 'test_pos_14', 'test_pos_15', 'test_pos_16', 'test_pos_17', 'test_pos_18', 'test_pos_19', 'test_pos_20', 'test_pos_21', 'test_pos_22', 'test_pos_23', 'test_pos_24', 'test_pos_25', 'test_pos_26', 'test_pos_27', 'test_pos_28', 'test_pos_29', 'test_pos_30', 'test_pos_31', 'test_pos_32', 'test_pos_33', 'test_pos_34', 'test_pos_35', 'test_pos_36', 'test_pos_37', 'test_pos_38', 'test_pos_39', 'test_pos_40', 'test_pos_41', 'test_pos_42', 'test_pos_43', 'test_pos_44', 'test_pos_45', 'test_pos_46', 'test_pos_47', 'test_pos_48', 'test_pos_49', 'test_pos_50', 'test_pos_51', 'test_pos_52', 'test_pos_53', 'test_pos_54', 'test_pos_55', 'test_pos_56', 'test_pos_57', 'test_pos_58', 'test_pos_59', 'test_pos_60', 'test_pos_61', 'test_pos_62', 'test_pos_63', 'test_pos_64', 'test_pos_65', 'test_pos_66', 'test_pos_67', 'test_pos_68', 'test_pos_69', 'test_pos_70', 'test_pos_71', 'test_pos_72', 'test_pos_73', 'test_pos_74', 'test_pos_75', 'test_pos_76', 'test_pos_77', 'test_pos_78', 'test_pos_79', 'test_pos_80', 'test_pos_81', 'test_pos_82', 'test_pos_83', 'test_pos_84', 'test_pos_85', 'test_pos_86', 'test_pos_87', 'test_pos_88', 'test_pos_89', 'test_pos_90', 'test_pos_91', 'test_pos_92', 'test_pos_93', 'test_pos_94', 'test_pos_95', 'test_pos_96', 'test_pos_97', 'test_pos_98', 'test_pos_99', 'test_pos_100', 'test_pos_101', 'test_pos_102', 'test_pos_103', 'test_pos_104', 'test_pos_105', 'test_pos_106', 'test_pos_107', 'test_pos_108', 'test_pos_109', 'test_pos_110', 'test_pos_111', 'test_pos_112', 'test_pos_113', 'test_pos_114', 'test_pos_115', 'test_pos_116', 'test_pos_117', 'test_pos_118', 'test_pos_119', 'test_pos_120', 'test_pos_121', 'test_pos_122', 'test_pos_123', 'test_pos_124', 'test_pos_125', 'test_pos_126', 'test_pos_127', 'test_pos_128', 'test_pos_129', 'test_pos_130', 'test_pos_131', 'test_pos_132', 'test_pos_133', 'test_pos_134', 'test_pos_135', 'test_pos_136', 'test_pos_137', 'test_pos_138', 'test_pos_139', 'test_pos_140', 'test_pos_141', 'test_pos_142', 'test_pos_143', 'test_pos_144', 'test_pos_145', 'test_pos_146', 'test_pos_147', 'test_pos_148', 'test_pos_149', 'test_pos_150', 'test_pos_151', 'test_pos_152', 'test_pos_153', 'test_pos_154']
test_smi = ['c1c(oc(c1)CN(C)C)CSCCNc1nc(c(c[nH]1)Cc1cc2c(cc1)cccc2)=O', 'c1c(c(ncc1)CSCCNC(=[NH]C#N)NCC)Br', 'n1c(csc1[NH]=C(N)N)c1cccc(c1)N', 'n1c(csc1[NH]=C(N)N)c1cccc(c1)NC(C)=O', 'n1c(csc1[NH]=C(N)N)c1cccc(c1)NC(NC)=[NH]C#N', 'CC(C)NCC(O)COc1ccc(CC(N)=O)cc1', 'OC4=C(C1CCC(CC1)c2ccc(Cl)cc2)C(=O)c3ccccc3C4=O', 'C(C1C(NC(Cc2c[nH]cn2)C(NC(C(NC(C(=O)NC(C(NC(C(NC(C(N1)=O)C(CC)C)=O)CCCN)=O)CCCCNC(C(NC(C(NC(C(NC(C1N=C(C(C(CC)C)N)SC1)=O)CC(C)C)=O)CCC(O)=O)=O)C(CC)C)=O)CC(N)=O)=O)CC(O)=O)=O)=O)c1ccccc1', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(c2c(CC1)[nH]cn2)CN1CCCC1)=O', 'CC(C)CC7N5C(=O)C(NC(=O)C2CN(C)C1Cc3c(Br)[nH]c4cccc(C1=C2)c34)(OC5(O)C6CCCN6C7=O)C(C)C', 'NS(=O)(=O)c2cc1c(N=CNS1(=O)=O)cc2Cl', 'CC34CC(O)C1C(CCC2=CC(=O)CCC12C)C3CCC4C(=O)CO', 'CC13CCC(=O)C=C1CCC4C2CCC(O)(C(=O)CO)C2(C)CC(O)C34', 'CC(CN1CC(=O)NC(=O)C1)N2CC(=O)NC(=O)C2', 'OCC1CCC(O1)n2cnc3c(=O)[nH]cnc23', 'CN1CC(CC2C1Cc3c[nH]c4cccc2c34)C(=O)NC8(C)OC7(O)C5CCCN5C(=O)C(Cc6ccccc6)N7C8=O', 'c1ccc(C(CN2CCC(C2)O)N(C(Cc2ccccc2N)=O)C)cc1', 'CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N2CCCC2C(O)=O', 'CCC(CO)NCCNC(CC)CO', 'CC(=O)N2CCN(C(CN1CCC(O)C1)C2)C(=O)Cc3ccc(Cl)c(Cl)c3', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(C2)O)=O)ccc(C(F)(F)F)cc1', 'c1(ccc(cc1)SC)CC(N1C(CN(CC1)C(=O)C)CN1CCC(C1)O)=O', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(C2)O)=O)cc(cc(c1)F)F', 'CC(=O)N2CCN(C(CN1CCC(O)C1)C2)C(=O)Cc3ccccc3', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(O)C2)=O)ccc(N(=O)=O)cc1', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(C2)O)=O)ccc(OC)cc1', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(O)C2)=O)cc(ccc1)N(=O)=O', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(O)C2)=O)ccc(S(=O)C)cc1', 'c1(CC(N2C(CN(CC2)C(=O)C)CN2CCC(O)C2)=O)ccc(S(=O)(=O)C)cc1', 'NC(=N)NCC2COC1(CCCCC1)O2', 'COc1cccnc1CCCCNc3nc(=O)c(Cc2ccc(C)nc2)c[nH]3', 'OC(=O)C1CCn2c1ccc2C(=O)c3ccccc3', 'CCn2cc(C(O)=O)c(=O)c3cc(F)c(N1CCNC(C)C1)c(F)c23', 'CN(C)Cc3ccc(CSCCNc2nc(=O)c(Cc1ccc(C)nc1)c[nH]2)o3', 'OCC(O)C(O)C(O)C(O)CO', 'CC(N)C(O)c1cccc(O)c1', 'CC#CC3(O)CCC4C2CCC1=CC(=O)CCC1=C2C(CC34C)c5ccc(cc5)N(C)C', 'NCC(O)c1ccc(O)c(O)c1', 'Nc2ccc(N=Nc1ccccc1)c(N)n2', 'CC3(C)SC2C(NC(=O)COc1ccccc1)C(=O)N2C3C(O)=O', 'CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(O)=O)C12', 'c12c([C@H]([C@H]3[N@@]4C[C@H](C=C)[C@H](C3)CC4)O)ccnc1ccc(c2)OC', 'CNC(NCCSCc1ccc(CN(C)C)o1)=CN(=O)=O', 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CC2(CC1)NC(NC2=O)=O)CN1CCCC1)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN2CC(CC2)O)c2n(CC1)ccn2)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN2CC(CC2)O)CN(CC1)CCO)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN2CC(CC2)O)c2c(CC1)[nH]cn2)=O', 'c1cc2c(C(NC(CC)c3ccccc3)=O)c(c(nc2cc1)c1ccccc1)OCC(O)=O', 'c1cc2c(C(NC(CC)c3ccccc3)=O)c(c(c3ccccc3)nc2cc1)OCCCC(O)=O', 'c1ccc2c(c(C(NC(CC)c3ccccc3)=O)c(OCCNC(Cc3c(cccc3)C(=O)O)=O)c(c3ccccc3)n2)c1', 'c1cc2c(c(OCCNC(Cc3ncccc3)=O)c(c3ccccc3)nc2cc1)C(NC(CC)c1ccccc1)=O', 'c1cc2c(c(c(c3ccccc3)nc2cc1)OCCNC(C1CCCN1)=O)C(=O)NC(CC)c1ccccc1', 'c1cc2c(C(NC(CC)c3ccccc3)=O)c(c(nc2cc1)c1ccccc1)Cn1cncc1', 'CC1CN(CC(C)N1)c4c(F)c(N)c3c(=O)c(cn(C2CC2)c3c4F)C(O)=O', 'CN(N=O)C(=O)NC1C(O)OC(CO)C(O)C1O', 'Cc3ccc(Cc2c[nH]c(NCCCCc1ncc(Br)cc1C)nc2=O)cn3', 'CN2C(=C(O)c1sccc1S2(=O)=O)C(=O)Nc3ccccn3', 's1cc(nc1[NH]=C(N)N)C', 'c1cc(ncc1)CSCCNc1c(cc[nH]1)N(=O)=O', 'c1(cc(nc(c1C)C)C(SC(CNc1c(cc[nH]1)N(=O)=O)(C)C)(C)C)C', 'n1c(csc1[NH]=C(N)N)c1ccccc1', 'C1CN(CCC1)Cc1cccc(c1)OCCCNC(=O)C', 'C1CN(CCC1)Cc1cccc(c1)OCCCNC(=O)c1ccccc1', 'C1CN(CCC1)Cc1cccc(c1)OCCCO', 'C1CN(CCC1)Cc1cccc(c1)OCCCNc1ncccc1', 'C1CN(CCC1)Cc1cccc(c1)OCCCNc1nccs1', 'C1CN(CCC1)Cc1cccc(c1)OCCCNc1nc2c(o1)cccc2', 'CCCN(CCC)CCc1ccc(c2c1CC(N2)=C)O', 'c1cc(CCNC)ncc1', 'c1cc(CCN(C)C)ncc1', 's1c(ncc1)CCN', 'c1nc(C2CCN(CC2)C(NC2CCCCC2)=S)c[nH]1', 's1cc(CSCCNC(NC)=[NH]C#N)nc1[NH]=C(N)N', 'CC(Cl)(Cl)Cl', 'C(CCl)(F)(F)F', 'n12c3c(C(c4c(cccc4)F)=NCc1cnc2CO)cc(Cl)cc3', 'CCC(C)(C)C', 'CC(CCC)C', 'CC(C)(O)C(F)(F)F', 'CC(C)O', 'CCCC(C)CC', 'CCC(C)CC', 'c12c(C(c3c(cccc3)F)=NC(c3n1c(nc3)C)O)cc(Cl)cc2', 'c1ccc2Oc3c(cc(cc3)Cl)C3C(c2c1)CN(CC3)C', 'c1c(Cl)nc(N2CCN(CCCCN3C(CCC3)=O)CC2)cc1C(F)(F)F', 'c1ccc2Oc3c(cc(cc3)Cl)C3C(c2c1)CNCC3', 'n1(c(c2nc[nH]c2n(c1=O)C)=O)C', 'c1c2CCOc2c(cc1OC)CNC1CCCNC1c1ccccc1', 'c1(cc(NC(=[NH]c2cccc(c2)CC)C)ccc1)CC', 'c1(c(cc2n(ncc2c1)CCN)Cl)Cl', 'c1(c2c(cc(F)cc2)on1)C1CCN(CCc2c(n3c(C(CCC3)O)nc2C)=O)CC1', 'Nc4nc(NC1CC1)c3ncn(C2CC(CO)C=C2)c3n4', 'CCCSc2ccc1[nH]c(NC(=O)OC)nc1c2', 'O=c1[nH]cnc2[nH]ncc12', 'CN(C)c2c(C)n(C)n(c1ccccc1)c2=O', 'Cc2cc(=O)n(c1ccccc1)n2C', 'c1(NCCSCc2oc(cc2)CN(C)C)[nH]cc(c1N(=O)=O)Cc1ccccc1', 'c1ccccc1', 'CC(C)(C)OC(=O)c1ncn3c1C2CCCN2C(=O)c4cc(Br)ccc34', 'c1(cc(c(cc1)Cl)Cl)CC(N1CCCCC1CN1CCCC1)=O', 'c1(cc(c(cc1)Cl)Cl)CC(N1C(c2c(CC1)cccc2)CN1CCCC1)=O', 'c1(ccc(cc1)C(F)(F)F)CC(N1C(CCCC1)CN1CCCC1)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(c2c(CC1)scc2)CN1CCCC1)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(c2c(CC1)ccs2)CN1CCCC1)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN2CCCC2)CC(CC1)(C)C)=O', 'CS(=O)(=O)OCCCCOS(C)(=O)=O', 'CCC(C)=O', 'Cn1cnc2n(C)c(=O)n(C)c(=O)c12', 'c12C3C(c4c(cccc4)N(c1cccc2)C(N)=O)O3', 'ClCCNC(=O)N(CCCl)N=O', 'COC2CN(CCCOc1ccc(F)cc1)CCC2NC(=O)c3cc(Cl)c(N)cc3OC', 'Nc1nc(Cl)nc2n(cnc12)C3CC(O)C(CO)O3', 'Clc1cccc(Cl)c1N=C2NCCN2', 'c1cn(c(c(c1=O)O)C)C', 'c1cn(CC)c(c(c1=O)O)C', 'c1cn(CCCC)c(C)c(O)c1=O', 'c1cn(CCCCCC)c(C)c(O)c1=O', 'c1cn(CCCCC)c(C)c(O)c1=O', 'c1cn(CC)c(CC)c(c1=O)O', 'C1CCCCC1', 'NC1CONC1=O', 'Nc2ccn(C1OC(CO)C(O)C1O)c(=O)n2', 'CN(C)N=Nc1[nH]cnc1C(N)=O', 'Clc1ccc2NC(=O)CC(=O)N(c3ccccc3)c2c1', 'N1(c2c(CCc3c1cccc3)cccc2)CCCN', 'Clc2cccc3NC(=O)CN=C(c1ccccc1)c23', 'N1(c2c(Sc3c1cccc3)cccc2)CCCNC', 'CCO', 'CCc1cc(ccn1)C(N)=S', 'CCc1ccccc1', 'NC(N)=Nc1nc(CSCCC(N)=NS(N)(=O)=O)cs1', 'CCOC(=O)c1ncn2c1CN(C)C(=O)c3cc(F)ccc23', 'P(C(=O)O)(=O)(O)O', 'c1(cc(c(cc1)Cl)Cl)CC(N1C(CN2CCCC2)CN(CC1)C(=O)C)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(c2c(CC1)occ2)CN1CCCC1)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN(CC1)C(=O)OC)CN1CCCC1)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN(CC1)C(=O)OCC)CN1CCCC1)=O', 'c1(CC(N2C(CN(CC2)C(=O)OCCC)CN2CCCC2)=O)cc(c(cc1)Cl)Cl', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(c2c(CC1)occ2)CN1CC(CC1)O)=O', 'CCCCCCC', 'CCCCCC', 'C(NO)(N)=O', 'CC(C)Cc1ccc(cc1)C(C)C(O)=O', 'CC(C)C(CN1CCCC1)N(C)C(=O)Cc2ccc(Cl)c(Cl)c2', 'CN(C(CN1CCCC1)c2ccccc2)C(=O)Cc3ccc(Cl)c(Cl)c3', 'NNC(=O)c1ccncc1', 'c12c(C(N(Cc3n1cnc3c1noc(n1)C(C)C)C)=O)c(ccc2)Cl', 'ClCCN(N=O)C(=O)NC1CCCCC1', 'OC(C1CCCCN1)c2cc(nc3c(cccc23)C(F)(F)F)C(F)(F)F', 'COc2ccc(CN(CCN(C)C)c1ccccn1)cc2', 'CC1CCCC1', 'COCCc1ccc(OCC(O)CNC(C)C)cc1', 'Cc1cccc(C)c1', 'OC1C=CC2C3Cc4ccc(O)c5OC1C2(CCN3CC=C)c45', 'Cc3ccnc4N(C1CC1)c2ncccc2C(=O)Nc34', 'N1(c2c(Sc3c1cccc3)ccc(c2)Cl)CCCNC', 'N1(c2c(Sc3c1cccc3)ccc(c2)Cl)CCCN', 'N1(c2c(Sc3c1cccc3)ccc(c2)SC)CCC1NCCCC1', 'C1NCCN(C1)c1ccc(c(n1)Cl)C(F)(F)F', 'C1N(CCN(C1)c1cc(ccn1)C(F)(F)F)CCCCN1C(CCC1)=O', 'c1c2c(cc(c1)Cl)C1C(c3ccccc3O2)CNC1', 'c1c2c(ccc1)C1(C(c3cccc(c3O2)C)CNCC1)O', 'c1ccc(c(c1)C(CC=C)N)c1c2c(on1)cccc2', 'CN1CC3C(C1)c2cc(Cl)ccc2Oc4ccccc34', 'Cc1ccccc1C', 'O=c1c2c([nH]c(=O)n1C)ncn2C', 'CCCCC', 'OCC1OC(CC1O)n2cnc3C(O)CNC=Nc23', 'c12C3(C(N(C)CC3)N(c1ccc(c2)OC(Nc1ccccc1)=O)C)C', 'CCCCC1C(=O)N(N(C1=O)c2ccccc2)c3ccccc3', 'CC(C)NCC(O)COc1cccc2[nH]ccc12', 'CN1CCN(CC1)CC(=O)N3c2ccccc2C(=O)Nc4cccnc34', 'COC(C5Cc4cc3cc(OC2CC(OC1CC(O)C(O)C(C)O1)C(O)C(C)O2)c(C)c(O)c3c(O)c4C(=O)C5OC8CC(OC7CC(O)C(OC6CC(C)(O)C(O)C(C)O6)C(C)O7)C(O)C(C)O8)C(=O)C(O)C(C)O', 'O=C1CN(CC2N1CCc3ccccc23)C(=O)C4CCCCC4', 'CC(=O)C3CCC4C2CCC1=CC(=O)CCC1(C)C2CCC34C', 'CCCO', 'CC(C)=O', 'Cc1ccc(C)cc1', 'NC(=O)c1cnccn1', 'CCc1nc(N)nc(N)c1c2ccc(Cl)cc2', 'c12c([C@@H]([C@@H]3[N@@]4C[C@H](C=C)[C@H](C3)CC4)O)ccnc1ccc(c2)OC', 'NC(=O)c1ncn(n1)C2OC(CO)C(O)C2O', 'CN3Cc1c(ncn1c2ccsc2C3=O)C(=O)OC(C)(C)C', 'Oc4ccc3[nH]cc(CCCCN1CCC(=CC1)c2ccccc2)c3c4', 'c1ccc2N(C(CN3CCCC3)C)c3c(ccc(C(=O)NCCC)c3)Sc2c1', 'OC(=O)CNC(=O)c1ccccc1O', 'c1(ccc(c(c1)Cl)Cl)CC(=O)N1C(CN(C)CC1)CN1CCCC1', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN2CCCC2)c2n(CC1)ccn2)=O', 'c1(ccc(c(c1)Cl)Cl)CC(N1C(CN2CCCC2)c2n(CC1)ncn2)=O', 'c12c(nc(c(c1C(NC(CC)c1ccccc1)=O)C)c1ccccc1)cccc2', 'c1ccc2nc(c(c(C(NC(CC)c3ccccc3)=O)c2c1)O)c1ccccc1', 'c1cc2c(C(NC(CC)c3ccccc3)=O)c(c(nc2cc1)c1ccccc1)OCCCN(C)C', 'CCCN(CCC)CCc1ccc(O)c2NC(=O)Cc12', 'CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1', 'Cc2cn(C1OC(CO)C=C1)c(=O)[nH]c2=O', 'c1(S(Nc2c(c(C)no2)C)(=O)=O)ccc(N)cc1', 'Nc2c1CCCCc1nc3ccccc23', 'c1cc(ccc1CCCC(OC(C)(C)C)=O)N(CCCl)CCCl', 'CC34CCC1C(CCC2=CC(=O)CCC12C)C3CCC4O', 'CCCC(C)C1(CC)C(=O)NC(=S)NC1=O', 'S=P(N1CC1)(N2CC2)N3CC3', 'CC3CC1=C(CCC(=O)C1)C4CCC2(C)C(CCC2(O)C#C)C34', 'CN3C(CNC(=O)c1ccsc1)CN=C(c2ccccc2F)c4ccccc34', 'CNC(NCCSCc1csc(N=C(N)N)n1)=NC#N', 'Cc1ccccc1', 'C(=C)(Cl)(Cl)Cl', 'NC1C2CN(CC12)c4nc3n(cc(C(O)=O)c(=O)c3cc4F)c5ccc(F)cc5F', 'CN(C1CCCCC1N2CCCC2)C(=O)Cc3ccc(Cl)c(Cl)c3', 'NC(N)=O', 'Nc1ncnc2n(cnc12)C3OC(CO)C(O)C3O', 'Nc2ccn(C1CCC(CO)O1)c(=O)n2', 'C1CCN(CC1)Cc4cccc(OCCCNc3nc2ccccc2s3)c4']
y_true = [0 if 'pos' in id else 1 for id in test_id]
BASE_DIR = os.path.abspath('.')
data_path = os.path.join(BASE_DIR, 'datapkl')
model_path = os.path.join(BASE_DIR, 'models')
base_data_file = os.path.join(data_path, 'mixed_img_tab_text.pkl')
test_data_file = os.path.join(data_path, 'test-data.pkl')

deep_models = ['tab', 'img', 'text', 'tab_img', 'tab_text', 'img_text', 'deep-b3']


def find_best_cutoff(fpr, tpr, thresholds):
    y = tpr-fpr
    youden_index = np.nanargmax(y)
    cutoff = thresholds[youden_index]
    return cutoff


def run_predict(model, data):
    learn = load_learner(model_path, model)
    learn.data = data
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    preds = preds.numpy()[:, 0]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, preds, pos_label=0)
    cutoff = find_best_cutoff(fpr, tpr, thresholds)
    y_pred = list(map(lambda x: 0 if x > cutoff else 1, preds))
    conf = metrics.confusion_matrix(y_true, y_pred)
    tp = conf[0, 0]
    fn = conf[0, 1]
    fp = conf[1, 0]
    tn = conf[1, 1]
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    logging.info('the predict results')
    logging.info('sn\t{0}'.format(sn))
    logging.info('sp\t{0}'.format(sp))
    logging.info('acc\t{0}'.format(acc))
    logging.info('auc\t{0}'.format(auc))
    logging.info('mcc\t{0}'.format(mcc))
    return None


def predict(model):
    base_file = open(base_data_file, 'rb')
    base_data = pickle.load(base_file)
    base_file.close()
    test_file = open(test_data_file, 'rb')
    test_data = pickle.load(test_file)
    test_file.close()

    base_data.add_test(test_data)
    if model == 'all':
        for m in deep_models:
            logging.info('begin run the model {0}'.format(m))
            modelpth = '{0}.pth'.format(m)
            run_predict(modelpth, base_data)
    else:
        logging.info('begin run the model {0}'.format(model))
        modelpth = '{0}.pth'.format(model)
        run_predict(modelpth, base_data)

def parse_args():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(prog='Deep-B3', description=description)
    parser.add_argument('-model',
                        required=True,
                        choices=['tab', 'img', 'text', 'tab_img', 'tab_text', 'img_text', 'deep-b3', 'all'],
                        help="test model for choice")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = args.model
    predict(model)
