from dataclasses import dataclass

@dataclass
class BboxesConfig:
    data_path = "/local/scratch/CaDISv2/"
    output_path = "../data/BoundingBoxes"

    classes = [
        'background',
        'Pupil',
        'Surgical Tape',
        'Hand',
        'Eye Retractors',
        'Iris',
        'Skin',
        'Cornea',
        'Hydrodissection Cannula',
        'Viscoelastic Cannula',
        'Capsulorhexis Cystotome',
        'Rycroft Cannula',
        'Bonn Forceps',
        'Primary Knife',
        'Phacoemuslsifier Handpiece',
        'Lens Injector',
        'I/A Handpiece',
        'Secondary Knife',
        'Micromanipulator',
        'I/A Handpiece Handle',
        'Capsulorhexis Forceps',
        'Rycroft Cannula Handle',
        'Phacoemulsifier Handpiece Handle',
        'Capsulorhexis Cystotome Handle',
        'Secondary Knife Handle',
        'Lens Injector Handle',
        'Suture Needle',
        'Needle Holder',
        'Charleux Cannula',
        'Primary Knife Handle',
        'Vitrectomy Handpiece',
        'Mendez Ring',
        'Marker',
        'Hydrodissection Cannula Handle',
        'Troutman Forceps',
        'Cotton',
        'Iris Hook'
    ]

    videos = [
        'Video01',
        'Video02',
        'Video03',
        'Video04',
        'Video05',
        'Video06',
        'Video07',
        'Video08',
        'Video09',
        'Video10',
        'Video11',
        'Video12',
        'Video13',
        'Video14',
        'Video15',
        'Video16',
        'Video17',
        'Video18',
        'Video19',
        'Video20',
        'Video21',
        'Video22',
        'Video23',
        'Video24',
        'Video25',
    ]

config = BboxesConfig()