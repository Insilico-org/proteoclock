import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class IPFPathways:
    """Define core IPF-related pathways and their constituent proteins"""

    def __init__(self):
        self.pathways = {
            'TGF_BETA': ["ACVR1", "APC", "ARID4B", "BCAR3", "BMP2", "BMPR1A", "BMPR2",
                         "CDH1", "CDK9", "CDKN1C", "CTNNB1", "ENG", "FKBP1A", "FNTA",
                         "FURIN", "HDAC1", "HIPK2", "ID1", "ID2", "ID3", "IFNGR2",
                         "JUNB", "KLF10", "LEFTY2", "LTBP2", "MAP3K7", "NCOR2", "NOG",
                         "PMEPA1", "PPM1A", "PPP1CA", "PPP1R15A", "RAB31", "RHOA",
                         "SERPINE1", "SKI", "SKIL", "SLC20A1", "SMAD1", "SMAD3",
                         "SMAD6", "SMAD7", "SMURF1", "SMURF2", "SPTBN1", "TGFB1",
                         "TGFBR1", "TGIF1", "THBS1", "TJP1", "TRIM33", "UBE2D3", "WWTR1", "XIAP"],
            'ECM_REMODELING': ["ABI3BP", "ACTA2", "ADAM12", "ANPEP", "APLP1", "AREG", "BASP1", "BDNF",
                               "BGN", "BMP1", "CADM1", "CALD1", "CALU", "CAP2", "CAPG", "CCN1", "CCN2",
                               "CD44", "CD59", "CDH11", "CDH2", "CDH6", "COL11A1", "COL12A1", "COL16A1",
                               "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL5A1", "COL5A2",
                               "COL5A3", "COL6A2", "COL6A3", "COL7A1", "COL8A2", "COLGALT1", "COMP",
                               "COPA", "CRLF1", "CTHRC1", "CXCL1", "CXCL12", "CXCL6", "CXCL8",
                               "DAB2", "DCN", "DKK1", "DPYSL3", "DST", "ECM1", "ECM2", "EDIL3",
                               "EFEMP2", "ELN", "EMP3", "ENO2", "FAP", "FAS", "FBLN1", "FBLN2",
                               "FBLN5", "FBN1", "FBN2", "FERMT2", "FGF2", "FLNA", "FMOD", "FN1",
                               "FOXC2", "FSTL1", "FSTL3", "FUCA1", "FZD8", "GADD45A", "GADD45B",
                               "GAS1", "GEM", "GJA1", "GLIPR1", "GPC1", "GPX7", "GREM1", "HTRA1",
                               "ID2", "IGFBP2", "IGFBP3", "IGFBP4", "IL15", "IL32", "IL6",
                               "INHBA", "ITGA2", "ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGB5",
                               "JUN", "LAMA1", "LAMA2", "LAMA3", "LAMC1", "LAMC2", "LGALS1",
                               "LOX", "LOXL1", "LOXL2", "LRP1", "LRRC15", "LUM", "MAGEE1",
                               "MATN2", "MATN3", "MCM7", "MEST", "MFAP5", "MGP", "MMP1",
                               "MMP14", "MMP2", "MMP3", "MSX1", "MXRA5", "MYL9", "MYLK", "NID2",
                               "NNMT", "NOTCH2", "NT5E", "NTM", "OXTR", "P3H1", "PCOLCE",
                               "PCOLCE2", "PDGFRB", "PDLIM4", "PFN2", "PLAUR", "PLOD1", "PLOD2",
                               "PLOD3", "PMEPA1", "PMP22", "POSTN", "PPIB", "PRRX1", "PRSS2",
                               "PTHLH", "PTX3", "PVR", "QSOX1", "RGS4", "RHOB", "SAT1", "SCG2",
                               "SDC1", "SDC4", "SERPINE1", "SERPINE2", "SERPINH1", "SFRP1",
                               "SFRP4", "SGCB", "SGCD", "SGCG", "SLC6A8", "SLIT2", "SLIT3",
                               "SNAI2", "SNTB1", "SPARC", "SPOCK1", "SPP1", "TAGLN", "TFPI2",
                               "TGFB1", "TGFBI", "TGFBR3", "TGM2", "THBS1", "THBS2", "THY1",
                               "TIMP1", "TIMP3", "TNC", "TNFAIP3", "TNFRSF11B", "TNFRSF12A",
                               "TPM1", "TPM2", "TPM4", "VCAM1", "VCAN", "VEGFA", "VEGFC",
                               "VIM", "WIPF1", "WNT5A"],
            'INFLAMMATION': ["ABCA1", "ABI1", "ACVR1B", "ACVR2A", "ADGRE1", "ADM",
                             "ADORA2B", "ADRM1", "AHR", "APLNR", "AQP9", "ATP2A2",
                             "ATP2B1", "ATP2C1", "AXL", "BDKRB1", "BEST1", "BST2",
                             "BTG2", "C3AR1", "C5AR1", "CALCRL", "CCL17", "CCL2",
                             "CCL20", "CCL22", "CCL24", "CCL5", "CCL7", "CCR7",
                             "CCRL2", "CD14", "CD40", "CD48", "CD55", "CD69", "CD70",
                             "CD82", "CDKN1A", "CHST2", "CLEC5A", "CMKLR1", "CSF1",
                             "CSF3", "CSF3R", "CX3CL1", "CXCL10", "CXCL11", "CXCL6",
                             "CXCL8", "CXCL9", "CXCR6", "CYBB", "DCBLD2", "EBI3", "EDN1",
                             "EIF2AK2", "EMP3", "EREG", "F3", "FFAR2", "FPR1", "FZD5", "GABBR1",
                             "GCH1", "GNA15", "GNAI3", "GP1BA", "GPC3", "GPR132", "GPR183", "HAS2",
                             "HBEGF", "HIF1A", "HPN", "HRH1", "ICAM1", "ICAM4", "ICOSLG", "IFITM1",
                             "IFNAR1", "IFNGR2", "IL10", "IL10RA", "IL12B", "IL15", "IL15RA",
                             "IL18", "IL18R1", "IL18RAP", "IL1A", "IL1B", "IL1R1", "IL2RB",
                             "IL4R", "IL6", "IL7R", "INHBA", "IRAK2", "IRF1", "IRF7", "ITGA5",
                             "ITGB3", "ITGB8", "KCNA3", "KCNJ2", "KCNMB2", "KIF1B", "KLF6", "LAMP3",
                             "LCK", "LCP2", "LDLR", "LIF", "LPAR1", "LTA", "LY6E", "LYN", "MARCO", "MEFV",
                             "MEP1A", "MET", "MMP14", "MSR1", "MXD1", "MYC", "NAMPT", "NDP", "NFKB1",
                             "NFKBIA", "NLRP3", "NMI", "NMUR1", "NOD2", "NPFFR2", "OLR1", "OPRK1", "OSM",
                             "OSMR", "P2RX4", "P2RX7", "P2RY2", "PCDH7", "PDE4B", "PDPN", "PIK3R5",
                             "PLAUR", "PROK2", "PSEN1", "PTAFR", "PTGER2", "PTGER4", "PTGIR", "PTPRE",
                             "PVR", "RAF1", "RASGRP1", "RELA", "RGS1", "RGS16", "RHOG", "RIPK2",
                             "RNF144B", "ROS1", "RTP4", "SCARF1", "SCN1B", "SELE", "SELENOS", "SELL",
                             "SEMA4D", "SERPINE1", "SGMS2", "SLAMF1", "SLC11A2", "SLC1A2", "SLC28A2",
                             "SLC31A1", "SLC31A2", "SLC4A4", "SLC7A1", "SLC7A2", "SPHK1", "SRI", "STAB1",
                             "TACR1", "TACR3", "TAPBP", "TIMP1", "TLR1", "TLR2", "TLR3", "TNFAIP6",
                             "TNFRSF1B", "TNFRSF9", "TNFSF10", "TNFSF15", "TNFSF9", "TPBG", "VIP"],
            'OXIDATIVE_STRESS': ["ABCC1", "ATOX1", "CAT", "CDKN2D", "EGLN2", "ERCC2", "FES",
                                 "FTL", "G6PD", "GCLC", "GCLM", "GLRX", "GLRX2", "GPX3",
                                 "GPX4", "GSR", "HHEX", "HMOX2", "IPCEF1", "JUNB", "LAMTOR5",
                                 "LSP1", "MBP", "MGST1", "MPO", "MSRA", "NDUFA6", "NDUFB4",
                                 "NDUFS2", "NQO1", "OXSR1", "PDLIM1", "PFKP", "PRDX1", "PRDX2",
                                 "PRDX4", "PRDX6", "PRNP", "PTPA", "SBNO2", "SCAF4", "SELENOS",
                                 "SOD1", "SOD2", "SRXN1", "STK25", "TXN", "TXNRD1", "TXNRD2"]
        }

    def get_pathway_genes(self, pathway_name):
        return self.pathways.get(pathway_name, [])

    def get_all_pathways(self):
        return self.pathways

class ScaleShift(nn.Module):
    def __init__(self, scale=85, shift=40):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return x * self.scale + self.shift

class UKBDataset(torch.utils.data.Dataset):

    def __init__(self, olink_data: np.ndarray, ages: np.ndarray):
        self.olink_data = torch.FloatTensor(olink_data)
        self.ages = torch.FloatTensor(ages)
        # self.ages = torch.FloatTensor(np.log2(ages))

    def __len__(self):
        return len(self.olink_data)

    def __getitem__(self, idx):
        return {
            'olink': self.olink_data[idx],
            'age': self.ages[idx]
        }

    # def denormalize_age(self, log_age):
    #     return 2 ** log_age

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Initialize weights considering log-scale input
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            m.bias.data.fill_(np.log2(65))  # Initialize bias around log2 of mean age

# Data utility functions for loading test data
import importlib.resources
import io
from pathlib import Path

def load_test_age_data():
    """
    Load the sample age data included with the proteoclock package.
    
    Returns:
        pandas.DataFrame: DataFrame containing age values for test samples
    """
    with importlib.resources.files("proteoclock.materials.test_data").joinpath("age_values.txt").open('rb') as f:
        return pd.read_csv(io.BytesIO(f.read()), sep='\t', header=0, index_col=None)

def load_test_protein_data():
    """
    Load the sample protein data included with the proteoclock package.
    
    Returns:
        pandas.DataFrame: DataFrame containing protein measurements in the format
                         required by proteoclock (patient_id, gene_symbol, NPX)
    """
    with importlib.resources.files("proteoclock.materials.test_data").joinpath("test_run_olink_data.tsv").open('rb') as f:
        return pd.read_csv(io.BytesIO(f.read()), sep='\t', header=0, index_col=None)
        
def load_kuo_test_age_data():
    """
    Load the Kuo et al. sample age data included with the proteoclock package.
    
    Returns:
        pandas.DataFrame: DataFrame containing age values for Kuo test samples
    """
    with importlib.resources.files("proteoclock.materials.test_data").joinpath("kuo_age_values.txt").open('rb') as f:
        return pd.read_csv(io.BytesIO(f.read()), sep='\t', header=0, index_col=None)
        
def load_kuo_test_protein_data():
    """
    Load the Kuo et al. sample protein data included with the proteoclock package.
    
    Returns:
        pandas.DataFrame: DataFrame containing protein measurements for Kuo test data
    """
    with importlib.resources.files("proteoclock.materials.test_data").joinpath("kuo_test_data.tsv").open('rb') as f:
        return pd.read_csv(io.BytesIO(f.read()), sep='\t', header=0, index_col=None)
