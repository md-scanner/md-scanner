import random
import pandas as pd
from os import path
from classify import ClassifyFontStyle
from torchvision.io import read_image
import numpy as np
import torch
import sys
from os import environ as env


# A list of English words generated using nltk
WORD_ARRAY = ['detacher', 'neffy', 'kaimo', 'nana', 'fess', 'objector', 'lydian', 'kamasin', 'dyaus', 'auditory', 'scruffy', 'adjudge', 'wrapper', 'closure', 'dendrite', 'fanman', 'vadimony', 'buzylene', 'pandowdy', 'precess', 'african', 'gambogic', 'poppet', 'reastray', 'scuddy', 'lofty', 'sobranje', 'ogreism', 'volvulus', 'deimos', 'porkwood', 'samburu', 'keddah', 'extort', 'visitant', 'detail', 'fustet', 'synapses', 'hillsman', 'itoubou', 'tarry', 'coxite', 'woodpeck', 'talk', 'repenter', 'sanies', 'turnkey', 'vaned', 'colder', 'textman', 'sport', 'roque', 'agal', 'himwards', 'dispose', 'flatdom', 'aides', 'ailing', 'charmer', 'spicery', 'mousse', 'dandle', 'stephan', 'overfear', 'sworn', 'hurri', 'copehan', 'nankin', 'ungassed', 'wounder', 'sinsion', 'vivarium', 'fermery', 'bond', 'galloman', 'sthenia', 'koklas', 'unswayed', 'osmotic', 'bevel', 'trimodal', 'latchet', 'barrable', 'augurate', 'toxodon', 'cursus', 'moki', 'baguette', 'ramplor', 'trental', 'rut', 'kalamian', 'wyke', 'cosecant', 'dada', 'demurely', 'lessener', 'degerm', 'domanial', 'misdraw', 'hypha', 'fanal', 'paynim', 'querken', 'reagin', 'danaide', 'uigur', 'leverer', 'tammie', 'chicness', 'pursy', 'aladfar', 'pennaria', 'subtread', 'logcock', 'walloper', 'moonack', 'benzylic', 'humility', 'khan', 'papmeat', 'latinity', 'jungli', 'croceic', 'tricklet', 'gritless', 'ramate', 'meso', 'vortices', 'sandyish', 'haoma', 'discal', 'ambrette', 'becloak', 'artie', 'nebbuk', 'abassin', 'halbert', 'steelify', 'maintain', 'katalase', 'jilt', 'tayir', 'saker', 'circle', 'drony', 'unnobly', 'markmoot', 'sphagion', 'inviter', 'mordva', 'andiroba', 'daemony', 'tingler', 'zincky', 'renverse', 'nirvana', 'almida', 'termon', 'zeta', 'seljuk', 'platerer', 'serfhood', 'adenine', 'narcissi', 'astutely', 'vilayet', 'imposure', 'oenocyte', 'wasoga', 'cystitis', 'dummel', 'rand', 'rebasis', 'hallman', 'joyweed', 'woning', 'tongas', 'nataraja', 'elinor', 'flicky', 'bluebead', 'teeny', 'statice', 'camphine', 'exosperm', 'sithcund', 'biennium', 'foliaged', 'searcer', 'toader', 'usings', 'skieppe', 'pilled', 'sprag', 'teedle', 'dar', 'walapai', 'fainness', 'acology', 'crosser', 'wastable', 'intitule', 'flot', 'cothy', 'dithion', 'cite', 'descent', 'coutil', 'linework', 'cutch', 'cordwood', 'deluder', 'dob', 'scaum', 'minorage', 'hotch', 'refloor', 'uproad', 'pouncing', 'deaden', 'upspout', 'kill', 'marinade', 'notioned', 'newcal', 'dinky', 'benedict', 'vannai', 'beballed', 'nullism', 'loom', 'buckle', 'proudly', 'pomerium', 'topazite', 'zodiacal', 'plugged', 'prearm', 'fusilier', 'orbitele', 'refinery', 'bookie', 'bhadon', 'peart', 'demiram', 'metreta', 'derek', 'airport', 'ephod', 'sportula', 'fleysome', 'demidog', 'kernish', 'attirer', 'rescuer', 'lombard', 'corky', 'unchewed', 'goatskin', 'reinform', 'chiasmal', 'feteless', 'retag', 'catvine', 'worn', 'comely', 'evict', 'steroid', 'skink', 'cepa', 'deedeed', 'azimech', 'pawnshop', 'ginners', 'hawaiite', 'unduke', 'stackman', 'overmeek', 'euclea', 'metaphor', 'huey', 'dokmarok', 'cathexis', 'guanaco', 'enemy', 'outsing', 'shawnee', 'arbustum', 'gaddi', 'remover', 'cheaply', 'cocullo', 'yardsman', 'hoistway', 'tanzeb', 'glucemia', 'zealotic', 'repiner', 'apotype', 'vasewise', 'nonfocal', 'taffrail', 'uninured', 'groomish', 'subfix', 'suberone', 'fugler', 'squame', 'beek', 'bard', 'disperse', 'promerit', 'padraic', 'lazily', 'ulcerate', 'naebody', 'writee', 'pulvinic', 'rheeboc', 'abscind', 'roxie', 'ping', 'emeline', 'potlike', 'tadousac', 'tjanting', 'wallower', 'milicent', 'becost', 'cahot', 'educt', 'positive', 'piercel', 'hybodus', 'rabbonim', 'scratch', 'ushabti', 'aswooned', 'prionops', 'inequity', 'absurdly', 'witchman', 'genoa', 'dreader', 'shard', 'crawdad', 'bryology', 'septime', 'mimus', 'parthian', 'redepend', 'rhamnite', 'unseam', 'samen', 'blacking', 'tacca', 'screak', 'slating', 'burgeon', 'enapt', 'shake', 'untaut', 'regrind', 'squabbed', 'dazzler', 'crossarm', 'porthors', 'plankton', 'deltic', 'yogi', 'platicly', 'sculler', 'dapifer', 'prawny', 'battery', 'neossin', 'damlike', 'blinker', 'majoon', 'suckable', 'synochus', 'kamaloka', 'notably', 'enspell', 'nutlet', 'sperma', 'gilding', 'sideway', 'pringle', 'trallian', 'garth', 'asaprol', 'proddle', 'orgiasm', 'wrist', 'garbel', 'uptube', 'overtide', 'jagir', 'finific', 'sade', 'disrank', 'raucid', 'dactylar', 'tinware', 'chronist', 'anergy', 'bunion', 'casaun', 'xeroma', 'codpiece', 'palolo', 'rondelet', 'foveate', 'termitic', 'unsavory', 'lauric', 'pompier', 'encrinal', 'voidee', 'mooneye', 'swallo', 'rosabel', 'tunga', 'hosel', 'balija', 'grannom', 'giggly', 'pagoda', 'innuit', 'unjilted', 'ole', 'fleck', 'exarchal', 'urushiye', 'estadal', 'pomona', 'floorman', 'genep', 'brushet', 'watching', 'warori', 'lobber', 'sinew', 'uaraycu', 'eyestone', 'peccancy', 'hylist', 'chitchat', 'ladyship', 'plussage', 'fanatic', 'abipon', 'gusty', 'cherty', 'lawsonia', 'antu', 'zoilist', 'besteer', 'drilling', 'outsole', 'venesia', 'puffinus', 'shaivism', 'samovar', 'sought', 'parazoan', 'parcel', 'flirter', 'incivic', 'gavel', 'penful', 'chorten', 'estop', 'sybarite', 'weirdly', 'toruloid', 'unnipped', 'ortol', 'chiripa', 'piki', 'cattleya', 'blitz', 'khoja', 'offhand', 'lolo', 'achromat', 'sepaline', 'deicer', 'cannot', 'intrude', 'clifty', 'altruist', 'styxian', 'ritling', 'ambeer', 'murder', 'toiletry', 'kinkle', 'sundang', 'wega', 'elegant', 'isoscele', 'pannus', 'serphid', 'afghani', 'yahoodom', 'cooba', 'pussycat', 'pyruvyl', 'pungence', 'stroud', 'family', 'pigmaker', 'cand', 'taxation', 'notehead', 'thuggish', 'begreen', 'unwelted', 'alexis', 'trickish', 'taborer', 'bacterin', 'millpost', 'poticary', 'liberty', 'eaves', 'chello', 'scraunch', 'cardin', 'clype', 'umbo', 'trichia', 'bepaw', 'puny', 'eternize', 'mutative', 'treculia', 'mycosis', 'inship', 'stupend', 'atelets', 'clary', 'routhy', 'essenis', 'graian', 'virose', 'quesited', 'sheetage', 'mistaker', 'testacea', 'purfle', 'rogan', 'outbud', 'apaid', 'lynch', 'perigyny', 'manship', 'haycock', 'ferrado', 'remock', 'macaw', 'typhonic', 'teck', 'stokesia', 'myoxidae', 'tach', 'malate', 'edacity', 'handful', 'zolle', 'panchama', 'aphthoid', 'fluxroot', 'hashab', 'siphonal', 'kaladana', 'jacobic', 'petit', 'gadget', 'slitless', 'foambow', 'lording', 'soloth', 'coat', 'franco', 'agama', 'pierdrop', 'frimaire', 'lioness', 'parcook', 'folder', 'solenite', 'menopoma', 'myotomy', 'pahi', 'get', 'tuik', 'campana', 'snorer', 'wilding', 'nicotia', 'kassite', 'unwifely', 'torinese', 'bebusy', 'lancepod', 'nabalism', 'deaconry', 'sheugh', 'pectinic', 'humulene', 'pettle', 'ovolo', 'biforked', 'savvy', 'mamers', 'vanadium', 'bourtree', 'rubicola', 'vacuous', 'lapicide', 'outsaint', 'gentes', 'dichter', 'nun', 'snouted', 'plate', 'bugseed', 'bepuddle', 'raman', 'cyzicene', 'locrine', 'pushful', 'caza', 'yird', 'outwake', 'fairling', 'sosoish', 'refuse', 'mackle', 'undashed', 'dataria', 'sposh', 'connect', 'stabwort', 'bryan', 'euryale', 'liknon', 'tromp', 'prostate', 'unjewel', 'unreel', 'sceat', 'forkful', 'gangrel', 'palpal', 'arecaine', 'cleche', 'jingly', 'magani', 'epacmaic', 'bander', 'sapples', 'justen', 'fixable', 'chlamyd', 'uromyces', 'axwise', 'scarfy', 'puppis', 'zephyr', 'topiary', 'ewder', 'sailship', 'leanness', 'uncomfy', 'subiya', 'unfound', 'credenza', 'dipteryx', 'cheque', 'nascan', 'hydroid', 'kenogeny', 'dosage', 'homey', 'needle', 'kalidium', 'bad', 'oxyether', 'chelys', 'frigate', 'anandria', 'rotator', 'taking', 'plower', 'pivot', 'chiliomb', 'douar', 'unturned', 'fired', 'nonusage', 'dahabeah', 'tallwood', 'unthread', 'immoral', 'schimmel', 'endless', 'aeonian', 'alberto', 'acylate', 'lacery', 'jube', 'anton', 'crandall', 'tabitude', 'cozener', 'peroxide', 'orbific', 'frankify', 'tereu', 'dispeace', 'mucose', 'deranged', 'bathos', 'mirdaha', 'broidery', 'uncini', 'esoteric', 'impurity', 'locker', 'noiseful', 'dramshop', 'isleward', 'nuzzle', 'engaze', 'dogmouth', 'yerth', 'foison', 'amianth', 'royal', 'enteria', 'codiaeum', 'semidry', 'piglet', 'mozzetta', 'desition', 'spikelet', 'barvel', 'rundlet', 'afebrile', 'acaulous', 'sludder', 'walker', 'endable', 'salema', 'hawky', 'mispick', 'wur', 'crepis', 'dismask', 'analyzer', 'zoned', 'ungnaw', 'debit', 'boreiad', 'yokeable', 'keeping', 'unbolt', 'afeared', 'chekist', 'tapa', 'afear', 'ambos', 'bifolia', 'warcraft', 'pathless', 'quipsome', 'acrisius', 'odorator', 'archgod', 'thrinter', 'unemploy', 'hygric', 'haikwan', 'alkenyl', 'isodont', 'awat', 'traduce', 'sharada', 'yoicks', 'resinify', 'heraldic', 'pagehood', 'bolivian', 'smectic', 'upbelt', 'chinband', 'uskara', 'huddle', 'clayton', 'flogster', 'saktism', 'folial', 'ramus', 'lepra', 'snaking', 'nodulize', 'cacomixl', 'zygosis', 'apigenin', 'chkalik', 'orphrey', 'gousty', 'tepid', 'jesse', 'currier', 'fanion', 'tierer', 'muggins', 'divorcer', 'rosetted', 'peesoreh', 'clutch', 'blazy', 'graded', 'arzun', 'boutylka', 'subduce', 'oresteia', 'myocyte', 'ranidae', 'unsash', 'didactyl', 'aliipoe', 'chevron', 'silver', 'rakh', 'apachite', 'windfall', 'estamp', 'nepote', 'gyne', 'canid', 'enscroll', 'signifer', 'poggy', 'sinuitis', 'kwapa', 'kiskatom', 'gateward', 'rocketry', 'grivna', 'skybal', 'ischium', 'somewhy', 'fogless', 'clubfoot', 'spiegel', 'papless', 'ballmine', 'catawba', 'porphyra', 'paintpot', 'swaglike', 'wheyish', 'valetism', 'ensuance', 'cofaster', 'bandwork', 'nidal', 'menurae', 'sinusal', 'moy', 'unhuman', 'harbor', 'schene', 'fust', 'wither', 'foolery', 'shi', 'sericate', 'exactor', 'mouthily', 'beneath', 'aerocyst', 'whipworm', 'atheris', 'uncrown', 'niobium', 'success', 'unglaze', 'saidi', 'waking', 'sai', 'stroil', 'iambist', 'chera', 'terebene', 'theow', 'rubidium', 'laten', 'lambskin', 'furazane', 'belgic', 'wiglike', 'matawan', 'chockman', 'cotonam', 'girse', 'tivy', 'ulla', 'murinus', 'fetch', 'talahib', 'hest', 'sound', 'sleazy', 'elfland', 'willie', 'manto', 'abridge', 'haustrum', 'grafter', 'unbulled', 'yokelry', 'cunonia', 'iodic', 'gameful', 'thof', 'khasi', 'atour', 'lai', 'areito', 'mulberry', 'unhooper', 'pettyfog', 'ripping', 'artamus', 'untone', 'beshod', 'healful', 'unsocial', 'pejorist', 'rolliche', 'otalgia', 'jingo', 'suricate', 'andrenid', 'accepted', 'lipeurus', 'unchoked', 'upheap', 'kapok', 'podler', 'thymotic', 'lide', 'unabased', 'retiring', 'blart', 'shaggily', 'tilery', 'drafty', 'fougade', 'windling', 'trowing', 'mistetch', 'zarema', 'airward', 'gaggle', 'stage', 'conduct', 'xeriff', 'cocamama', 'doricism', 'tursio', 'drupe', 'flogger', 'papyrine', 'fumitory', 'pelter', 'havildar', 'dunite', 'tommy', 'inkfish', 'nicaean', 'pickwork', 'plasome', 'overtalk', 'sofane', 'rizzle', 'milvus', 'xylinid', 'tragal', 'ossarium', 'scarcen', 'nontoxic']


NUM_SAMPLES = 1000
MIN_BATCH_SIZE = 128


dataset = pd.read_csv(env['FSC_DATASET_CSV_PATH'])


def _filter_dataset(is_italic: bool, is_bold: bool):
    return dataset[(dataset['is_italic'] == is_italic) & (dataset['is_bold'] == is_bold)]


def _calc_precision_recall_f1(tp, tn, fp, fn) -> float:
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * (p * r) / (p + r)
    return p, r, f1


def _encode_style(italic: bool, bold: bool) -> int:
    if italic: # Italic
        return 1
    elif bold: # Bold
        return 2
    else: # Regular
        return 0
    

def _decode_style(style: int) -> bool:
    # TODO we need this function because in the dataset we have separate bools for italic/bold, we're
    # not using an index for the style
    return [
        (False, False), # Regular
        (True, False),  # Italic
        (False, True)   # Bold
    ][style]


def eval_word_classification(style: int):
    batch = []
    ground_truth = []
    word_refs = []

    word_idx = 0

    tp, tn, fp, fn = 0, 0, 0, 0

    while True:
        while len(batch) < MIN_BATCH_SIZE and word_idx < NUM_SAMPLES:
            word = random.choice(WORD_ARRAY)
            word_idx += 1

            # Pick a word belonging to the given `style` with a 50% probability
            pick_same_style = round(random.random()) >= 0.5
            if pick_same_style:
                picked_style = _decode_style(style)  # Pick the same style
            else:
                picked_style = _decode_style(int(style + random.random() + 1) % 3)  # Pick another style

            ground_truth += [_encode_style(*picked_style)]

            word_dataset = _filter_dataset(*picked_style)

            # TODO QUICK FIX
            # We want all word's characters to be in the font (we weirdly have some fonts without all characters)
            sampled_font = None
            fonts = word_dataset['font'].unique()
            random.shuffle(fonts)
            for font in fonts:
                rows = word_dataset[word_dataset['char'].isin(list(word))]
                if len(rows) >= len(set(word)):
                    sampled_font = font
                    break
            
            if sampled_font == None:
                print(f"ERROR: Couldn't find a font with all word characters: {word}")
                continue

            char_refs = []  # An array of arrays, telling where the words' characters are in the batch

            for char in word:
                sample_char = word_dataset[word_dataset['char'] == char].iloc[0]

                sample_img_path = path.join(env['FSC_DATASET_DIR'], sample_char["filename"])
                sample_img = read_image(sample_img_path)  # Load a ByteTensor
                sample_img = sample_img.type(torch.FloatTensor) / 255.0  # Convert to FloatTensor

                char_refs += [len(batch)]
                batch += [(sample_img, sample_char["char"])]

            word_refs.append((word, char_refs))
        

        # Once we picked enough characters or words, we classify the batch
        if len(batch) > 0:
            classify = ClassifyFontStyle(batch)
            char_classification_result = np.array(classify())
            word_classification_result = []

            for (word, char_refs) in word_refs:
                word_style_idx = np.argmax(np.bincount(char_classification_result[char_refs]))
                #print(f"Word: {word}, Refs: {char_refs}, Char clf: {char_classification_result[char_refs]}, Word clf: {word_style_idx}")
                word_classification_result.append(word_style_idx)

            ground_truth = np.array(ground_truth)
            word_classification_result = np.array(word_classification_result)

            #print("Ground truth", ground_truth)
            #print("Classification", word_classification_result)

            tp += (word_classification_result[ground_truth == style] == style).sum()
            tn += (word_classification_result[ground_truth != style] != style).sum()
            fp += (word_classification_result[ground_truth != style] == style).sum()
            fn += (word_classification_result[ground_truth == style] != style).sum()
            
            p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)

            print(f"[eval] Processed {word_idx}/{NUM_SAMPLES} words...")
            print(f"\tTP: {tp}")
            print(f"\tTN: {tn}")
            print(f"\tFP: {fp}")
            print(f"\tFN: {fn}")
            print(f"\tPrecision: {p:.3f}")
            print(f"\tRecall: {r:.3f}")
            print(f"\tF1-score: {f1:.3f}")

            batch = []
            ground_truth = []
            word_refs = []

        if word_idx >= NUM_SAMPLES: # Did we reach the end?
            break

    return tp, tn, fp, fn


def main():
    if len(sys.argv) != 2:
        print(f"Invalid syntax: {sys.argv[0]} <out-csv>")
        sys.exit(1)

    out_csv_file = open(sys.argv[1], "w")
    out_csv_file.write(f"Style, TP, TN, FP, FN, Precision, Recall, F1-score\n")

    print(f"-" * 96)
    print(f"Regular word classification")
    print(f"-" * 96)

    tp, tn, fp, fn = eval_word_classification(0)
    p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)
    out_csv_file.write(f"Regular, {tp}, {tn}, {fp}, {fn}, {p:.3f}, {r:.3f}, {f1:.3f}\n")

    print(f"-" * 96)
    print(f"Italic word classification")
    print(f"-" * 96)

    tp, tn, fp, fn = eval_word_classification(1)
    p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)
    out_csv_file.write(f"Italic, {tp}, {tn}, {fp}, {fn}, {p:.3f}, {r:.3f}, {f1:.3f}\n")

    print(f"-" * 96)
    print(f"Bold word classification")
    print(f"-" * 96)

    tp, tn, fp, fn = eval_word_classification(2)
    p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)
    out_csv_file.write(f"Bold, {tp}, {tn}, {fp}, {fn}, {p:.3f}, {r:.3f}, {f1:.3f}\n")


if __name__ == "__main__":
    main()

