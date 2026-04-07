import os

SAVE_DIR = "./knowledge_base"
os.makedirs(SAVE_DIR, exist_ok=True)

# 模拟医学知识条目（后续可替换为真实 PDF / PubMed 摘要）
documents = [
    {
        "filename": "melanoma.txt",
        "content": """Melanoma is a type of skin cancer that develops from melanocytes.
Risk factors include UV exposure, fair skin, family history, and multiple moles.
The ABCDE criteria help identify suspicious lesions: Asymmetry, Border irregularity,
Color variation, Diameter greater than 6mm, and Evolution over time.
Treatment options include surgical excision, immunotherapy, targeted therapy,
and radiation. Early detection significantly improves survival rates.
Dermoscopy is recommended for evaluation of pigmented skin lesions."""
    },
    {
        "filename": "basal_cell_carcinoma.txt",
        "content": """Basal cell carcinoma (BCC) is the most common form of skin cancer.
It arises from basal cells in the deepest layer of the epidermis.
BCC rarely metastasizes but can cause significant local destruction if untreated.
Common presentations include pearly or waxy bumps, flat flesh-colored lesions,
and bleeding or scabbing sores that heal and return.
Treatment includes surgical excision, Mohs surgery, cryotherapy,
topical medications such as imiquimod, and radiation therapy.
Sun protection is the primary prevention strategy."""
    },
    {
        "filename": "psoriasis.txt",
        "content": """Psoriasis is a chronic autoimmune condition that causes rapid skin cell buildup.
It results in scaling on the skin surface with inflammation and redness around scales.
Common types include plaque psoriasis, guttate, inverse, pustular, and erythrodermic.
Triggers include stress, infections, certain medications, and skin injuries.
Treatment options range from topical corticosteroids and vitamin D analogues
to phototherapy, methotrexate, cyclosporine, and biologic agents targeting
TNF-alpha, IL-17, and IL-23 pathways."""
    },
    {
        "filename": "eczema.txt",
        "content": """Atopic dermatitis (eczema) is a chronic inflammatory skin condition
characterized by dry, itchy, and inflamed skin.
It commonly begins in childhood and may persist into adulthood.
The condition is associated with other atopic diseases such as asthma and allergic rhinitis.
Management includes regular moisturizing, avoiding triggers, topical corticosteroids,
calcineurin inhibitors, and newer biologic treatments such as dupilumab.
Infection with Staphylococcus aureus is a common complication."""
    },
    {
        "filename": "seborrheic_keratosis.txt",
        "content": """Seborrheic keratosis is one of the most common noncancerous skin growths
seen in older adults. They appear as brown, black or light tan growths on the face,
chest, shoulders or back. The growths have a waxy, scaly, slightly elevated appearance.
Seborrheic keratoses are not contagious and are not related to sun exposure.
No treatment is necessary unless they become irritated or you dislike how they look.
Removal options include cryotherapy, curettage, and laser treatment."""
    },
    {
        "filename": "dermatofibroma.txt",
        "content": """Dermatofibroma is a common benign skin growth that most often appears
on the lower legs. It is firm to the touch and may be pink, grey, red or brown in color.
The lesion typically dimples inward when pinched, known as the dimple sign.
Dermatofibromas are harmless and usually require no treatment.
They are more common in women than in men and can occur at any age.
If removal is desired for cosmetic reasons, surgical excision is the standard approach."""
    },
    {
        "filename": "vascular_lesions.txt",
        "content": """Vascular lesions of the skin include a variety of conditions such as
hemangiomas, port-wine stains, spider angiomas, and cherry angiomas.
They result from abnormal blood vessels in or near the skin surface.
Hemangiomas are the most common benign vascular tumors of infancy.
Treatment depends on the type, size, and location of the lesion.
Options include observation, laser therapy, sclerotherapy, and surgical excision.
Pulsed dye laser is considered the gold standard for many vascular lesions."""
    },
]

for doc in documents:
    path = os.path.join(SAVE_DIR, doc["filename"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc["content"])

print(f"已生成 {len(documents)} 个知识库文档到 {SAVE_DIR}/")
for doc in documents:
    print(f"  ├── {doc['filename']}")