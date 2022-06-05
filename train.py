from fastai.vision import *
from fastai.tabular import *
from fastai.text import *
from fastai.callbacks import *
from sklearn import metrics

import numpy as np
import pandas as pd
import argparse
import os
import logging
import warnings
warnings.filterwarnings('ignore')

import model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

cont_names = ["MaxEStateIndex","MinEStateIndex","MaxAbsEStateIndex","MinAbsEStateIndex","qed","MolWt","HeavyAtomMolWt","ExactMolWt","NumValenceElectrons","NumRadicalElectrons","MaxPartialCharge","MinPartialCharge","MaxAbsPartialCharge","MinAbsPartialCharge","FpDensityMorgan1","FpDensityMorgan2","FpDensityMorgan3","BCUT2D_MWHI","BCUT2D_MWLOW","BCUT2D_CHGHI","BCUT2D_CHGLO","BCUT2D_LOGPHI","BCUT2D_LOGPLOW","BCUT2D_MRHI","BCUT2D_MRLOW","BalabanJ","BertzCT","Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v","HallKierAlpha","Ipc","Kappa1","Kappa2","Kappa3","LabuteASA","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA14","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","TPSA","EState_VSA1","EState_VSA10","EState_VSA11","EState_VSA2","EState_VSA3","EState_VSA4","EState_VSA5","EState_VSA6","EState_VSA7","EState_VSA8","EState_VSA9","VSA_EState1","VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5","VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9","FractionCSP3","HeavyAtomCount","NHOHCount","NOCount","NumAliphaticCarbocycles","NumAliphaticHeterocycles","NumAliphaticRings","NumAromaticCarbocycles","NumAromaticHeterocycles","NumAromaticRings","NumHAcceptors","NumHDonors","NumHeteroatoms","NumRotatableBonds","NumSaturatedCarbocycles","NumSaturatedHeterocycles","NumSaturatedRings","RingCount","MolLogP","MolMR","fr_Al_COO","fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH","fr_Ar_OH","fr_COO","fr_COO2","fr_C_O","fr_C_O_noCOO","fr_C_S","fr_HOCCN","fr_Imine","fr_NH0","fr_NH1","fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole","fr_SH","fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid","fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide","fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic","fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether","fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imidazole","fr_imide","fr_isocyan","fr_isothiocyan","fr_ketone","fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine","fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitro_arom_nonortho","fr_nitroso","fr_oxazole","fr_oxime","fr_para_hydroxylation","fr_phenol","fr_phenol_noOrthoHbond","fr_phos_acid","fr_phos_ester","fr_piperdine","fr_piperzine","fr_priamide","fr_prisulfonamd","fr_pyridine","fr_quatN","fr_sulfide","fr_sulfonamd","fr_sulfone","fr_term_acetylene","fr_tetrazole","fr_thiazole","fr_thiocyan","fr_thiophene","fr_unbrch_alkane","fr_urea","MACCSKyes0","MACCSKyes1","MACCSKyes2","MACCSKyes3","MACCSKyes4","MACCSKyes5","MACCSKyes6","MACCSKyes7","MACCSKyes8","MACCSKyes9","MACCSKyes10","MACCSKyes11","MACCSKyes12","MACCSKyes13","MACCSKyes14","MACCSKyes15","MACCSKyes16","MACCSKyes17","MACCSKyes18","MACCSKyes19","MACCSKyes20","MACCSKyes21","MACCSKyes22","MACCSKyes23","MACCSKyes24","MACCSKyes25","MACCSKyes26","MACCSKyes27","MACCSKyes28","MACCSKyes29","MACCSKyes30","MACCSKyes31","MACCSKyes32","MACCSKyes33","MACCSKyes34","MACCSKyes35","MACCSKyes36","MACCSKyes37","MACCSKyes38","MACCSKyes39","MACCSKyes40","MACCSKyes41","MACCSKyes42","MACCSKyes43","MACCSKyes44","MACCSKyes45","MACCSKyes46","MACCSKyes47","MACCSKyes48","MACCSKyes49","MACCSKyes50","MACCSKyes51","MACCSKyes52","MACCSKyes53","MACCSKyes54","MACCSKyes55","MACCSKyes56","MACCSKyes57","MACCSKyes58","MACCSKyes59","MACCSKyes60","MACCSKyes61","MACCSKyes62","MACCSKyes63","MACCSKyes64","MACCSKyes65","MACCSKyes66","MACCSKyes67","MACCSKyes68","MACCSKyes69","MACCSKyes70","MACCSKyes71","MACCSKyes72","MACCSKyes73","MACCSKyes74","MACCSKyes75","MACCSKyes76","MACCSKyes77","MACCSKyes78","MACCSKyes79","MACCSKyes80","MACCSKyes81","MACCSKyes82","MACCSKyes83","MACCSKyes84","MACCSKyes85","MACCSKyes86","MACCSKyes87","MACCSKyes88","MACCSKyes89","MACCSKyes90","MACCSKyes91","MACCSKyes92","MACCSKyes93","MACCSKyes94","MACCSKyes95","MACCSKyes96","MACCSKyes97","MACCSKyes98","MACCSKyes99","MACCSKyes100","MACCSKyes101","MACCSKyes102","MACCSKyes103","MACCSKyes104","MACCSKyes105","MACCSKyes106","MACCSKyes107","MACCSKyes108","MACCSKyes109","MACCSKyes110","MACCSKyes111","MACCSKyes112","MACCSKyes113","MACCSKyes114","MACCSKyes115","MACCSKyes116","MACCSKyes117","MACCSKyes118","MACCSKyes119","MACCSKyes120","MACCSKyes121","MACCSKyes122","MACCSKyes123","MACCSKyes124","MACCSKyes125","MACCSKyes126","MACCSKyes127","MACCSKyes128","MACCSKyes129","MACCSKyes130","MACCSKyes131","MACCSKyes132","MACCSKyes133","MACCSKyes134","MACCSKyes135","MACCSKyes136","MACCSKyes137","MACCSKyes138","MACCSKyes139","MACCSKyes140","MACCSKyes141","MACCSKyes142","MACCSKyes143","MACCSKyes144","MACCSKyes145","MACCSKyes146","MACCSKyes147","MACCSKyes148","MACCSKyes149","MACCSKyes150","MACCSKyes151","MACCSKyes152","MACCSKyes153","MACCSKyes154","MACCSKyes155","MACCSKyes156","MACCSKyes157","MACCSKyes158","MACCSKyes159","MACCSKyes160","MACCSKyes161","MACCSKyes162","MACCSKyes163","MACCSKyes164","MACCSKyes165","MACCSKyes166","Morgan0","Morgan1","Morgan2","Morgan3","Morgan4","Morgan5","Morgan6","Morgan7","Morgan8","Morgan9","Morgan10","Morgan11","Morgan12","Morgan13","Morgan14","Morgan15","Morgan16","Morgan17","Morgan18","Morgan19","Morgan20","Morgan21","Morgan22","Morgan23","Morgan24","Morgan25","Morgan26","Morgan27","Morgan28","Morgan29","Morgan30","Morgan31","Morgan32","Morgan33","Morgan34","Morgan35","Morgan36","Morgan37","Morgan38","Morgan39","Morgan40","Morgan41","Morgan42","Morgan43","Morgan44","Morgan45","Morgan46","Morgan47","Morgan48","Morgan49","Morgan50","Morgan51","Morgan52","Morgan53","Morgan54","Morgan55","Morgan56","Morgan57","Morgan58","Morgan59","Morgan60","Morgan61","Morgan62","Morgan63","Morgan64","Morgan65","Morgan66","Morgan67","Morgan68","Morgan69","Morgan70","Morgan71","Morgan72","Morgan73","Morgan74","Morgan75","Morgan76","Morgan77","Morgan78","Morgan79","Morgan80","Morgan81","Morgan82","Morgan83","Morgan84","Morgan85","Morgan86","Morgan87","Morgan88","Morgan89","Morgan90","Morgan91","Morgan92","Morgan93","Morgan94","Morgan95","Morgan96","Morgan97","Morgan98","Morgan99","Morgan100","Morgan101","Morgan102","Morgan103","Morgan104","Morgan105","Morgan106","Morgan107","Morgan108","Morgan109","Morgan110","Morgan111","Morgan112","Morgan113","Morgan114","Morgan115","Morgan116","Morgan117","Morgan118","Morgan119","Morgan120","Morgan121","Morgan122","Morgan123","Morgan124","Morgan125","Morgan126","Morgan127","Morgan128","Morgan129","Morgan130","Morgan131","Morgan132","Morgan133","Morgan134","Morgan135","Morgan136","Morgan137","Morgan138","Morgan139","Morgan140","Morgan141","Morgan142","Morgan143","Morgan144","Morgan145","Morgan146","Morgan147","Morgan148","Morgan149","Morgan150","Morgan151","Morgan152","Morgan153","Morgan154","Morgan155","Morgan156","Morgan157","Morgan158","Morgan159","Morgan160","Morgan161","Morgan162","Morgan163","Morgan164","Morgan165","Morgan166","Morgan167","Morgan168","Morgan169","Morgan170","Morgan171","Morgan172","Morgan173","Morgan174","Morgan175","Morgan176","Morgan177","Morgan178","Morgan179","Morgan180","Morgan181","Morgan182","Morgan183","Morgan184","Morgan185","Morgan186","Morgan187","Morgan188","Morgan189","Morgan190","Morgan191","Morgan192","Morgan193","Morgan194","Morgan195","Morgan196","Morgan197","Morgan198","Morgan199","Morgan200","Morgan201","Morgan202","Morgan203","Morgan204","Morgan205","Morgan206","Morgan207","Morgan208","Morgan209","Morgan210","Morgan211","Morgan212","Morgan213","Morgan214","Morgan215","Morgan216","Morgan217","Morgan218","Morgan219","Morgan220","Morgan221","Morgan222","Morgan223","Morgan224","Morgan225","Morgan226","Morgan227","Morgan228","Morgan229","Morgan230","Morgan231","Morgan232","Morgan233","Morgan234","Morgan235","Morgan236","Morgan237","Morgan238","Morgan239","Morgan240","Morgan241","Morgan242","Morgan243","Morgan244","Morgan245","Morgan246","Morgan247","Morgan248","Morgan249","Morgan250","Morgan251","Morgan252","Morgan253","Morgan254","Morgan255","Morgan256","Morgan257","Morgan258","Morgan259","Morgan260","Morgan261","Morgan262","Morgan263","Morgan264","Morgan265","Morgan266","Morgan267","Morgan268","Morgan269","Morgan270","Morgan271","Morgan272","Morgan273","Morgan274","Morgan275","Morgan276","Morgan277","Morgan278","Morgan279","Morgan280","Morgan281","Morgan282","Morgan283","Morgan284","Morgan285","Morgan286","Morgan287","Morgan288","Morgan289","Morgan290","Morgan291","Morgan292","Morgan293","Morgan294","Morgan295","Morgan296","Morgan297","Morgan298","Morgan299","Morgan300","Morgan301","Morgan302","Morgan303","Morgan304","Morgan305","Morgan306","Morgan307","Morgan308","Morgan309","Morgan310","Morgan311","Morgan312","Morgan313","Morgan314","Morgan315","Morgan316","Morgan317","Morgan318","Morgan319","Morgan320","Morgan321","Morgan322","Morgan323","Morgan324","Morgan325","Morgan326","Morgan327","Morgan328","Morgan329","Morgan330","Morgan331","Morgan332","Morgan333","Morgan334","Morgan335","Morgan336","Morgan337","Morgan338","Morgan339","Morgan340","Morgan341","Morgan342","Morgan343","Morgan344","Morgan345","Morgan346","Morgan347","Morgan348","Morgan349","Morgan350","Morgan351","Morgan352","Morgan353","Morgan354","Morgan355","Morgan356","Morgan357","Morgan358","Morgan359","Morgan360","Morgan361","Morgan362","Morgan363","Morgan364","Morgan365","Morgan366","Morgan367","Morgan368","Morgan369","Morgan370","Morgan371","Morgan372","Morgan373","Morgan374","Morgan375","Morgan376","Morgan377","Morgan378","Morgan379","Morgan380","Morgan381","Morgan382","Morgan383","Morgan384","Morgan385","Morgan386","Morgan387","Morgan388","Morgan389","Morgan390","Morgan391","Morgan392","Morgan393","Morgan394","Morgan395","Morgan396","Morgan397","Morgan398","Morgan399","Morgan400","Morgan401","Morgan402","Morgan403","Morgan404","Morgan405","Morgan406","Morgan407","Morgan408","Morgan409","Morgan410","Morgan411","Morgan412","Morgan413","Morgan414","Morgan415","Morgan416","Morgan417","Morgan418","Morgan419","Morgan420","Morgan421","Morgan422","Morgan423","Morgan424","Morgan425","Morgan426","Morgan427","Morgan428","Morgan429","Morgan430","Morgan431","Morgan432","Morgan433","Morgan434","Morgan435","Morgan436","Morgan437","Morgan438","Morgan439","Morgan440","Morgan441","Morgan442","Morgan443","Morgan444","Morgan445","Morgan446","Morgan447","Morgan448","Morgan449","Morgan450","Morgan451","Morgan452","Morgan453","Morgan454","Morgan455","Morgan456","Morgan457","Morgan458","Morgan459","Morgan460","Morgan461","Morgan462","Morgan463","Morgan464","Morgan465","Morgan466","Morgan467","Morgan468","Morgan469","Morgan470","Morgan471","Morgan472","Morgan473","Morgan474","Morgan475","Morgan476","Morgan477","Morgan478","Morgan479","Morgan480","Morgan481","Morgan482","Morgan483","Morgan484","Morgan485","Morgan486","Morgan487","Morgan488","Morgan489","Morgan490","Morgan491","Morgan492","Morgan493","Morgan494","Morgan495","Morgan496","Morgan497","Morgan498","Morgan499","Morgan500","Morgan501","Morgan502","Morgan503","Morgan504","Morgan505","Morgan506","Morgan507","Morgan508","Morgan509","Morgan510","Morgan511","Morgan512","Morgan513","Morgan514","Morgan515","Morgan516","Morgan517","Morgan518","Morgan519","Morgan520","Morgan521","Morgan522","Morgan523","Morgan524","Morgan525","Morgan526","Morgan527","Morgan528","Morgan529","Morgan530","Morgan531","Morgan532","Morgan533","Morgan534","Morgan535","Morgan536","Morgan537","Morgan538","Morgan539","Morgan540","Morgan541","Morgan542","Morgan543","Morgan544","Morgan545","Morgan546","Morgan547","Morgan548","Morgan549","Morgan550","Morgan551","Morgan552","Morgan553","Morgan554","Morgan555","Morgan556","Morgan557","Morgan558","Morgan559","Morgan560","Morgan561","Morgan562","Morgan563","Morgan564","Morgan565","Morgan566","Morgan567","Morgan568","Morgan569","Morgan570","Morgan571","Morgan572","Morgan573","Morgan574","Morgan575","Morgan576","Morgan577","Morgan578","Morgan579","Morgan580","Morgan581","Morgan582","Morgan583","Morgan584","Morgan585","Morgan586","Morgan587","Morgan588","Morgan589","Morgan590","Morgan591","Morgan592","Morgan593","Morgan594","Morgan595","Morgan596","Morgan597","Morgan598","Morgan599","Morgan600","Morgan601","Morgan602","Morgan603","Morgan604","Morgan605","Morgan606","Morgan607","Morgan608","Morgan609","Morgan610","Morgan611","Morgan612","Morgan613","Morgan614","Morgan615","Morgan616","Morgan617","Morgan618","Morgan619","Morgan620","Morgan621","Morgan622","Morgan623","Morgan624","Morgan625","Morgan626","Morgan627","Morgan628","Morgan629","Morgan630","Morgan631","Morgan632","Morgan633","Morgan634","Morgan635","Morgan636","Morgan637","Morgan638","Morgan639","Morgan640","Morgan641","Morgan642","Morgan643","Morgan644","Morgan645","Morgan646","Morgan647","Morgan648","Morgan649","Morgan650","Morgan651","Morgan652","Morgan653","Morgan654","Morgan655","Morgan656","Morgan657","Morgan658","Morgan659","Morgan660","Morgan661","Morgan662","Morgan663","Morgan664","Morgan665","Morgan666","Morgan667","Morgan668","Morgan669","Morgan670","Morgan671","Morgan672","Morgan673","Morgan674","Morgan675","Morgan676","Morgan677","Morgan678","Morgan679","Morgan680","Morgan681","Morgan682","Morgan683","Morgan684","Morgan685","Morgan686","Morgan687","Morgan688","Morgan689","Morgan690","Morgan691","Morgan692","Morgan693","Morgan694","Morgan695","Morgan696","Morgan697","Morgan698","Morgan699","Morgan700","Morgan701","Morgan702","Morgan703","Morgan704","Morgan705","Morgan706","Morgan707","Morgan708","Morgan709","Morgan710","Morgan711","Morgan712","Morgan713","Morgan714","Morgan715","Morgan716","Morgan717","Morgan718","Morgan719","Morgan720","Morgan721","Morgan722","Morgan723","Morgan724","Morgan725","Morgan726","Morgan727","Morgan728","Morgan729","Morgan730","Morgan731","Morgan732","Morgan733","Morgan734","Morgan735","Morgan736","Morgan737","Morgan738","Morgan739","Morgan740","Morgan741","Morgan742","Morgan743","Morgan744","Morgan745","Morgan746","Morgan747","Morgan748","Morgan749","Morgan750","Morgan751","Morgan752","Morgan753","Morgan754","Morgan755","Morgan756","Morgan757","Morgan758","Morgan759","Morgan760","Morgan761","Morgan762","Morgan763","Morgan764","Morgan765","Morgan766","Morgan767","Morgan768","Morgan769","Morgan770","Morgan771","Morgan772","Morgan773","Morgan774","Morgan775","Morgan776","Morgan777","Morgan778","Morgan779","Morgan780","Morgan781","Morgan782","Morgan783","Morgan784","Morgan785","Morgan786","Morgan787","Morgan788","Morgan789","Morgan790","Morgan791","Morgan792","Morgan793","Morgan794","Morgan795","Morgan796","Morgan797","Morgan798","Morgan799","Morgan800","Morgan801","Morgan802","Morgan803","Morgan804","Morgan805","Morgan806","Morgan807","Morgan808","Morgan809","Morgan810","Morgan811","Morgan812","Morgan813","Morgan814","Morgan815","Morgan816","Morgan817","Morgan818","Morgan819","Morgan820","Morgan821","Morgan822","Morgan823","Morgan824","Morgan825","Morgan826","Morgan827","Morgan828","Morgan829","Morgan830","Morgan831","Morgan832","Morgan833","Morgan834","Morgan835","Morgan836","Morgan837","Morgan838","Morgan839","Morgan840","Morgan841","Morgan842","Morgan843","Morgan844","Morgan845","Morgan846","Morgan847","Morgan848","Morgan849","Morgan850","Morgan851","Morgan852","Morgan853","Morgan854","Morgan855","Morgan856","Morgan857","Morgan858","Morgan859","Morgan860","Morgan861","Morgan862","Morgan863","Morgan864","Morgan865","Morgan866","Morgan867","Morgan868","Morgan869","Morgan870","Morgan871","Morgan872","Morgan873","Morgan874","Morgan875","Morgan876","Morgan877","Morgan878","Morgan879","Morgan880","Morgan881","Morgan882","Morgan883","Morgan884","Morgan885","Morgan886","Morgan887","Morgan888","Morgan889","Morgan890","Morgan891","Morgan892","Morgan893","Morgan894","Morgan895","Morgan896","Morgan897","Morgan898","Morgan899","Morgan900","Morgan901","Morgan902","Morgan903","Morgan904","Morgan905","Morgan906","Morgan907","Morgan908","Morgan909","Morgan910","Morgan911","Morgan912","Morgan913","Morgan914","Morgan915","Morgan916","Morgan917","Morgan918","Morgan919","Morgan920","Morgan921","Morgan922","Morgan923","Morgan924","Morgan925","Morgan926","Morgan927","Morgan928","Morgan929","Morgan930","Morgan931","Morgan932","Morgan933","Morgan934","Morgan935","Morgan936","Morgan937","Morgan938","Morgan939","Morgan940","Morgan941","Morgan942","Morgan943","Morgan944","Morgan945","Morgan946","Morgan947","Morgan948","Morgan949","Morgan950","Morgan951","Morgan952","Morgan953","Morgan954","Morgan955","Morgan956","Morgan957","Morgan958","Morgan959","Morgan960","Morgan961","Morgan962","Morgan963","Morgan964","Morgan965","Morgan966","Morgan967","Morgan968","Morgan969","Morgan970","Morgan971","Morgan972","Morgan973","Morgan974","Morgan975","Morgan976","Morgan977","Morgan978","Morgan979","Morgan980","Morgan981","Morgan982","Morgan983","Morgan984","Morgan985","Morgan986","Morgan987","Morgan988","Morgan989","Morgan990","Morgan991","Morgan992","Morgan993","Morgan994","Morgan995","Morgan996","Morgan997","Morgan998","Morgan999","Morgan1000","Morgan1001","Morgan1002","Morgan1003","Morgan1004","Morgan1005","Morgan1006","Morgan1007","Morgan1008","Morgan1009","Morgan1010","Morgan1011","Morgan1012","Morgan1013","Morgan1014","Morgan1015","Morgan1016","Morgan1017","Morgan1018","Morgan1019","Morgan1020","Morgan1021","Morgan1022","Morgan1023"]
BASE_DIR = os.path.abspath('.')
pklpath = os.path.join(BASE_DIR, 'datapkl')
modelpath = os.path.join(BASE_DIR, 'models')

def set_random_seed(seed):
    import random
    random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    logging.info('deep-b3, set the random seed {0}'.format(seed))

def create_dirs(dir):
    path = os.path.join(BASE_DIR, dir)
    if not os.path.exists(path):
        logging.info('create a new dirs, {0}'.format(path))
        os.makedirs(path)
    return None

def read_csv_file(file):
    if not os.path.isfile(file):
        logging.error('{0} is not a file or not exists'.format(file))
        exit(-1)
    else:
        df = pd.read_csv(file)
        df.fillna(0, inplace=True)
        return df

def find_best_cutoff(fpr, tpr, thresholds):
    y = tpr-fpr
    youden_index = np.nanargmax(y)
    cutoff = thresholds[youden_index]
    return cutoff

def pretrain_nlp(train_df):
    bs = 8
    iter = 50
    smi = train_df[['label', 'smi']]
    data_lm = (TextList.from_df(
        smi, cols='smi'
    ).split_by_rand_pct(0.2).label_for_lm().databunch(bs=bs, path=Path(BASE_DIR)))
    nlp_lm_file = os.path.join(BASE_DIR, 'datapkl/{0}'.format('nlp_lm.pkl'))
    logging.info('save the nlp lm file {0}'.format(nlp_lm_file))
    data_lm.save(nlp_lm_file)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    callbacks = [
        EarlyStoppingCallback(learn, min_delta=1e-5, patience=4),
        SaveModelCallback(learn)
    ]
    logging.info('nlp pre-train model \n{0}'.format(learn.model))
    logging.info('{0}'.format(len(learn.layer_groups)))

    learn.freeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    lr_v = learn.recorder.min_grad_lr
    logging.info('find best learn rate {0}'.format(lr_v))
    learn.fit_one_cycle(iter, lr_v, callbacks=callbacks)
    freeze = len(learn.layer_groups)
    for ly in range(1, freeze + 1):
        ly = 0-ly
        learn.freeze_to(ly)
        learn.lr_find()
        learn.recorder.plot(suggestion=True)
        lr_v = learn.recorder.min_grad_lr
        logging.info('find best learn rate for freeze {0} {1}'.format(ly, lr_v))
        learn.fit_one_cycle(iter, lr_v, callbacks=callbacks)

    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    lr_v = learn.recorder.min_grad_lr
    logging.info('find best learn rate for unfreeze {0}'.format(lr_v))
    learn.fit_one_cycle(iter, lr_v, callbacks=callbacks)
    logging.info('save the encoder at models/text_encoder')
    learn.save_encoder('text_encoder')
    return None


def test_deep_b3(test_df):
    bs = test_df.shape[0]
    logging.info('new bs is {0}'.format(bs))
    modelfile = os.path.join(modelpath, 'deep-b3.pth')
    lmfile = os.path.join(pklpath, 'nlp_lm.pkl')
    trainfile = os.path.join(pklpath, 'data_train.pkl')
    if not os.path.isfile(modelfile):
        logging.error('deep-b3 model file not exists, please re-train it')
        exit(0)
    if not os.path.isfile(lmfile):
        logging.error('smiles nlp lm file not exists')
        exit(0)
    path = Path(BASE_DIR)
    data_lm = load_data(Path(pklpath), 'nlp_lm.pkl', bs=bs)
    vocab = data_lm.vocab
    procs = [FillMissing, Categorify, Normalize]

    imgListTest = ImageList.from_df(test_df, path=path, cols='PicturePath')
    tabListTest = TabularList.from_df(test_df, cat_names=[], cont_names=cont_names, procs=procs, path=path)
    textListTest = TextList.from_df(test_df, cols='smi', path=path, vocab=vocab)

    mixedTest = (MixedItemList([imgListTest, tabListTest, textListTest], path, inner_df=tabListTest.inner_df))

    train_file = open(trainfile, 'rb')
    data_train = pickle.load(train_file)
    train_file.close()
    data_train.add_test(mixedTest)
    data_train.batch_size = bs
    logging.info('the data train batch size is {0}'.format(data_train.batch_size))
    learnTest = load_learner(path=modelpath, file='deep-b3.pth')
    learnTest.data = data_train
    preds, y = learnTest.get_preds(ds_type=DatasetType.Test)
    pred = preds.numpy()[:, 0]

    #with open('pred_res.txt', 'w') as fw:
    #    for one in pred:
    #        fw.write(str(one) + '\n')

    y_true = test_df.label.to_list()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, pred, pos_label=0)
    y_pred = list(map(lambda x: 0 if x > 0.5 else 1, pred))
    conf = metrics.confusion_matrix(y_true, y_pred)
    logging.info('confusion matrix is {0}'.format(conf))
    tp = conf[0, 0]
    fn = conf[0, 1]
    fp = conf[1, 0]
    tn = conf[1, 1]
    logging.info('tp, fn, tn ,fp: {0},{1},{2},{3}'.format(tp, fn, tn, fp))
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    sp = tn / (tn + fp)
    sn = tp / (tp + fn)
    logging.info('auc is {0}'.format(auc))
    logging.info('acc is {0}'.format(acc))
    logging.info('mcc is {0}'.format(mcc))
    logging.info('sp is {0}'.format(sp))
    logging.info('sn is {0}'.format(sn))


def train_deep_b3(train_df, epoch, bs, vis_out, text_out, has_img, has_tab, has_text):
    path = Path(BASE_DIR)
    size = 224
    procs = [FillMissing, Categorify, Normalize]

    bytarget = train_df.groupby(['id', 'label']).size().reset_index()
    bytarget = bytarget.sample(frac=.2, random_state=2022).drop([0, 'label'], axis=1)
    bytarget['is_valid'] = True
    bbb_train = pd.merge(train_df, bytarget, how='left', on='id')
    bbb_train.is_valid = bbb_train.is_valid.fillna(False)

    if not os.path.exists(os.path.join(pklpath, 'nlp_lm.pkl')):
        logging.error('nlp lm data file not exists, please re-trained')
        exit(0)
    data_lm = load_data(Path(pklpath), 'nlp_lm.pkl', bs=bs)
    vocab = data_lm.vocab

    imgList = ImageList.from_df(bbb_train, path=path, cols='PicturePath')
    tabList = TabularList.from_df(bbb_train, cat_names=[], cont_names=cont_names, procs=procs, path=path)
    textList = TextList.from_df(bbb_train, cols='smi', path=path, vocab=vocab)
    mixed = (MixedItemList([imgList, tabList, textList], path, inner_df=tabList.inner_df)
        .split_from_df(col='is_valid')
        .label_from_df(cols='label')
        .transform([[get_transforms()[0], [], []], [get_transforms()[1], [], []]], size=size)
    )
    data = mixed.databunch(bs=bs, collate_fn=model.collate_mixed)
    norm, denorm = model.normalize_custom_funcs(*imagenet_stats)
    data.add_tfm(norm)  # normalize images

    logging.info('save the mixed data file')
    outfile = os.path.join(pklpath, 'data_train.pkl')
    outfile = open(outfile, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    
    data_text = (
        TextList.from_df(
            bbb_train, cols='smi', path=path, vocab=vocab
        )
    ).split_none(
    ).label_from_df(
        cols='label'
    ).databunch(bs=bs)

    learn = model.image_tabular_text_learner(
        data=data,
        len_cont_names=len(cont_names),
        nlp_cls=data_text,
        vis_out=vis_out,
        text_out=text_out,
        has_img=has_img,
        has_tab=has_tab,
        has_text=has_text,
        is_save=False
    )

    #loss_func = CrossEntropyFlat()
    #callbacks = [
    #    EarlyStoppingCallback(learn, min_delta=1e-5, patience=4),
    #    SaveModelCallback(learn)
    #]

    #learn.callbacks = callbacks
    # opt_func = partial(optim.Adam, lr=3e-5, betas=(0.9, 0.99), weight_decay=0.1, amsgrad=True)
    #learn.opt_func = opt_func

    logging.info('create multi-model \n {0}'.format(learn.model))
    logging.info('model summary  \n {0}'.format(learn.summary()))

    freeze = len(learn.layer_groups)
    logging.info('the number of layer is {0}'.format(freeze))

    learn.freeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True, skip=15)
    lr_v = learn.recorder.min_grad_lr
    logging.info('the best learn rate for multi-model is {0}'.format(lr_v))

    learn.fit_one_cycle(epoch, lr_v)
    for ly in range(1, freeze + 1):
        ly = 0 - ly
        learn.load('bestmodel')
        learn.freeze_to(ly)
        learn.lr_find()
        learn.recorder.plot(suggestion=True, skip=15)
        lr_v = learn.recorder.min_grad_lr
        logging.info('the best learn rate for freeze layer {0} is {1}'.format(ly, lr_v))
        learn.fit_one_cycle(epoch, lr_v)

    learn.load('bestmodel')
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True, skip=15)
    lr_v = learn.recorder.min_grad_lr
    logging.info('the best learn rate for unfreeze is {0}'.format(lr_v))
    learn.fit_one_cycle(epoch, lr_v)

    learn.load('bestmodel')
    exrpath = os.path.join(BASE_DIR, '{0}/{1}'.format('models', 'deep-b3.pth'))
    logging.info('export deep-b3 model as {0}'.format(exrpath))
    learn.export(exrpath)
    learn.save('mixed')

def str2bool(v):
    if isinstance(v, bool):
        return v
    return True if v.lower() == 'true' else False

def parse_args():
    parser = argparse.ArgumentParser(prog='Deep-B3')
    subparsers = parser.add_subparsers(dest="subcmd", help="Train or test the Deep-B3")

    parser_train = subparsers.add_parser('train', help='train a new Deep-B3')
    parser_test = subparsers.add_parser('test', help='test for Deep-B3')

    parser_train.add_argument('--feature', required=True, help='feature file for train model')
    parser_train.add_argument('--epoch', type=int, default=50, help="the epochs for each freeze layer to train")
    parser_train.add_argument('--bs', type=int, default=64, help="batch size")
    parser_train.add_argument('--has_img', type=str2bool, default='True', help="including img features")
    parser_train.add_argument('--has_tab', type=str2bool, default='True', help="including tab features")
    parser_train.add_argument('--has_text', type=str2bool, default='True', help="including text features")
    parser_train.add_argument('--vis_out', type=int, default=512, help="feature output from the CNN model for image")
    parser_train.add_argument('--text_out', type=int, default=64, help="feature output from the NLP model for SMILES")

    parser_test.add_argument('--feature', required=True, help='feature file for train model')

    args = parser.parse_args()
    if not args.subcmd:
        parser.print_help()
        exit(0)
    return args

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(2022)
    if args.subcmd == 'train':
        logging.info('begin training a new deep-b3 model')
        logging.info('vision outputs feature is {0}'.format(args.vis_out))
        logging.info('text outputs feature is {0}'.format(args.text_out))
        logging.info('batch size is {0}'.format(args.bs))
        logging.info('including image features ? [{0}]'.format(args.has_img))
        logging.info('including tabular features ? [{0}]'.format(args.has_tab))
        logging.info('including text features ? [{0}]'.format(args.has_text))
        create_dirs('datapkl')
        train = os.path.join(os.path.join(BASE_DIR, 'train'), args.feature)
        train_df = read_csv_file(train)

        train_df['PicturePath'] = train_df.id.map(lambda x: 'train_images/{0}.png'.format(x))

        train_df.smi = train_df.smi.map(lambda x: ' '.join(list(x)))
        nlp_encoder = os.path.join(modelpath, 'text_encoder.pth')
        if not os.path.exists(nlp_encoder):
            logging.info('nlp lm data file not exists, begin training nlp model')
            pretrain_nlp(train_df)
        train_deep_b3(
            train_df=train_df,
            epoch=int(args.epoch),
            bs=int(args.bs),
            vis_out=int(args.vis_out),
            text_out=int(args.text_out),
            has_img=args.has_img,
            has_tab=args.has_tab,
            has_text=args.has_text
        )

    if args.subcmd == 'test':
        test = os.path.join(os.path.join(BASE_DIR, 'test'), args.feature)
        test_df = read_csv_file(test)
        test_df['PicturePath'] = test_df.id.map(lambda x: 'test_images/{0}.png'.format(x))
        test_df.smi = test_df.smi.map(lambda x: ' '.join(list(x)))
        test_deep_b3(test_df)
