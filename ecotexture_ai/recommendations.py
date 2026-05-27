"""
EcoTexture AI — Recycling Knowledge Base
=========================================
Per-class recommendations, educational insights, and impact metrics.
Supports English and Arabic (bilingual for Egypt context).
"""
from __future__ import annotations

from ecotexture_ai.config import (
    ARABIC_LABELS, CO2_SAVINGS_KG, RECYCLING_BINS, SDG_TAGS, WASTE_CLASSES
)

# ─────────────────────────────────────────────────────────────
# RICH KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────
_KNOWLEDGE: dict[str, dict] = {
    "Cardboard": {
        "en": {
            "action":  "Flatten and place in the BLUE recycling bin. Remove any tape or staples.",
            "tip":     "One tonne of recycled cardboard saves 17 trees and 7,000 litres of water.",
            "insight": "Cardboard has up to 7 recycling cycles before fibres are too short to reuse.",
            "egypt":   "In Egypt, informal waste-pickers (Zabbaleen) have recycled cardboard for decades. Supporting formal recycling continues this vital work.",
        },
        "ar": {
            "action":  "اطوِ الكرتون واضعه في صندوق الإعادة الأزرق. أزل الشريط والدبابيس.",
            "tip":     "طن واحد من الكرتون المعاد تدويره يوفر 17 شجرة و 7000 لتر ماء.",
            "insight": "يمكن إعادة تدوير الكرتون حتى 7 مرات.",
            "egypt":   "الزبالون في مصر كانوا روادًا في إعادة تدوير الكرتون منذ عقود.",
        },
    },
    "Glass": {
        "en": {
            "action":  "Rinse and place in the GREEN bin. Separate by colour if possible.",
            "tip":     "Glass can be recycled infinitely without losing quality.",
            "insight": "Recycling glass cuts CO₂ emissions by 20% compared to virgin production.",
            "egypt":   "Egypt imports significant silica for glass manufacturing. Local recycling reduces this import dependency.",
        },
        "ar": {
            "action":  "اشطف الزجاج واضعه في الصندوق الأخضر.",
            "tip":     "يمكن إعادة تدوير الزجاج إلى ما لا نهاية دون فقدان الجودة.",
            "insight": "تقليل انبعاثات CO₂ بنسبة 20% مقارنة بإنتاج الزجاج الخام.",
            "egypt":   "مصر تستورد السيليكا لصناعة الزجاج؛ التدوير المحلي يقلل هذا الاستيراد.",
        },
    },
    "Metal": {
        "en": {
            "action":  "Rinse cans and place in the YELLOW bin. Aluminium and steel are both recyclable.",
            "tip":     "Recycling one aluminium can saves enough energy to power a TV for 3 hours.",
            "insight": "Aluminium recycling uses only 5% of the energy needed for primary smelting.",
            "egypt":   "Egypt's steel industry is growing. Scrap metal recycling supports local manufacturing and job creation.",
        },
        "ar": {
            "action":  "اشطف العلب واضعها في الصندوق الأصفر.",
            "tip":     "إعادة تدوير علبة ألمنيوم واحدة توفر طاقة تشغيل تلفاز 3 ساعات.",
            "insight": "يستهلك تدوير الألمنيوم 5% فقط من طاقة الصهر الأولي.",
            "egypt":   "صناعة الصلب المصرية تنمو؛ تدوير الخردة يدعم التصنيع المحلي.",
        },
    },
    "Organic": {
        "en": {
            "action":  "Place in the BROWN compost bin or home compost heap.",
            "tip":     "Composting organic waste cuts methane emissions from landfills by 50%.",
            "insight": "Organic waste is 55-60% of Egypt's municipal solid waste — composting it could transform urban agriculture.",
            "egypt":   "The Zabaleen community has pioneered composting in Cairo. Community composting hubs are expanding across Egyptian cities.",
        },
        "ar": {
            "action":  "ضع النفايات العضوية في الصندوق البني أو حاوية الكومبوست المنزلية.",
            "tip":     "الكومبوست يقلل انبعاثات الميثان من مدافن القمامة بنسبة 50%.",
            "insight": "النفايات العضوية تمثل 55-60% من نفايات مصر البلدية.",
            "egypt":   "مجتمع الزبالون رائد في الكومبوست بالقاهرة.",
        },
    },
    "Paper": {
        "en": {
            "action":  "Place clean, dry paper in the BLUE bin. Avoid wet, greasy, or coated paper.",
            "tip":     "Recycling 1 tonne of paper saves 24 trees and 26,000 litres of water.",
            "insight": "Egyptian paper consumption is rising with literacy rates — recycling every sheet counts.",
            "egypt":   "The Nile Paper Company is Egypt's largest consumer of recycled fibres. Your paper goes further than you think.",
        },
        "ar": {
            "action":  "ضع الورق النظيف والجاف في الصندوق الأزرق.",
            "tip":     "تدوير طن من الورق يوفر 24 شجرة و 26000 لتر ماء.",
            "insight": "استهلاك الورق في مصر يرتفع مع ارتفاع معدلات التعليم.",
            "egypt":   "شركة النيل للورق هي أكبر مستهلك للألياف المعاد تدويرها في مصر.",
        },
    },
    "Plastic_PET": {
        "en": {
            "action":  "Rinse, remove caps, and place in the YELLOW bin. PET (code 1) is highly recyclable.",
            "tip":     "A single PET bottle takes 450 years to decompose in a landfill.",
            "insight": "Recycled PET (rPET) is used in clothing, carpets, and new bottles, closing the loop.",
            "egypt":   "Egypt produces 1.5 million tonnes of plastic waste annually. PET recovery is critical to protect the Nile and Mediterranean coastline.",
        },
        "ar": {
            "action":  "اشطف الزجاجة وضعها في الصندوق الأصفر بعد إزالة الغطاء.",
            "tip":     "زجاجة PET واحدة تستغرق 450 عامًا للتحلل.",
            "insight": "rPET يُستخدم في الملابس والسجاد والزجاجات الجديدة.",
            "egypt":   "مصر تنتج 1.5 مليون طن من البلاستيك سنويًا.",
        },
    },
    "Plastic_HDPE": {
        "en": {
            "action":  "Place in the YELLOW bin. HDPE (code 2) — milk jugs, detergent bottles.",
            "tip":     "HDPE is one of the most recyclable plastics, with a strong secondary market.",
            "insight": "Recycled HDPE is used in drainage pipes, outdoor furniture, and playground equipment.",
            "egypt":   "HDPE recycling in Egypt supports the growing construction industry with sustainable materials.",
        },
        "ar": {
            "action":  "ضع في الصندوق الأصفر. HDPE (رمز 2) — علب الحليب وزجاجات المنظفات.",
            "tip":     "HDPE من أكثر البلاستيك قابلية للتدوير.",
            "insight": "يُستخدم HDPE المعاد تدويره في أنابيب الصرف والأثاث الخارجي.",
            "egypt":   "تدوير HDPE يدعم صناعة البناء في مصر.",
        },
    },
    "Plastic_PVC": {
        "en": {
            "action":  "Do NOT place in general recycling. Take to a specialist PVC collection point.",
            "tip":     "PVC (code 3) contains chlorine — burning or landfilling releases toxic dioxins.",
            "insight": "PVC recycling is technically complex; avoid single-use PVC where possible.",
            "egypt":   "Egypt's plumbing sector uses PVC extensively. Extended Producer Responsibility (EPR) schemes are needed.",
        },
        "ar": {
            "action":  "لا تضع في سلة إعادة التدوير العامة. خذه إلى نقطة جمع PVC المتخصصة.",
            "tip":     "PVC يحتوي على الكلور — حرقه يطلق ديوكسينات سامة.",
            "insight": "إعادة تدوير PVC معقدة تقنيًا.",
            "egypt":   "قطاع السباكة في مصر يستخدم PVC على نطاق واسع.",
        },
    },
    "Styrofoam": {
        "en": {
            "action":  "Place in the BLACK/general waste bin — most facilities cannot recycle Styrofoam.",
            "tip":     "Styrofoam (EPS) is 98% air but takes 500+ years to break down.",
            "insight": "Avoid Styrofoam packaging. Switch to paper or cornstarch alternatives.",
            "egypt":   "Egypt banned single-use plastic bags in 2021. A Styrofoam ban would be the logical next step.",
        },
        "ar": {
            "action":  "ضع في الصندوق الأسود — معظم المرافق لا تستطيع إعادة تدوير الفوم.",
            "tip":     "الفوم 98% هواء لكنه يستغرق 500 عام للتحلل.",
            "insight": "تجنب تعبئة الفوم. انتقل إلى بدائل الورق أو النشا.",
            "egypt":   "حظرت مصر الأكياس البلاستيكية أحادية الاستخدام عام 2021.",
        },
    },
    "Textile": {
        "en": {
            "action":  "Donate wearable clothing. Place worn textiles in PURPLE textile collection banks.",
            "tip":     "The fashion industry emits more CO₂ than aviation and shipping combined.",
            "insight": "Recycled textiles become insulation, industrial rags, and new fibres.",
            "egypt":   "Egypt is a major cotton producer. Textile recycling supports the circularity of its $3B textile export industry.",
        },
        "ar": {
            "action":  "تبرع بالملابس الصالحة. ضع النسيج المتهالك في صناديق التجميع البنفسجية.",
            "tip":     "صناعة الأزياء تصدر ثاني أكسيد الكربون أكثر من الطيران والشحن مجتمعَين.",
            "insight": "تُستخدم المنسوجات المعاد تدويرها في العزل والخيوط الجديدة.",
            "egypt":   "مصر منتج رئيسي للقطن. تدوير المنسوجات يدعم الاقتصاد الدائري لصناعة التصدير.",
        },
    },
    "E-Waste": {
        "en": {
            "action":  "Take to an authorised e-waste collection centre. NEVER bin electronics.",
            "tip":     "One million mobile phones contain 24 kg of gold, 250 kg of silver, and 9,000 kg of copper.",
            "insight": "E-waste is the fastest-growing waste stream globally, at 57 Mt/year.",
            "egypt":   "Egypt's e-waste generation is rising with smartphone penetration. The National E-Waste Programme (NEEP) needs citizen participation.",
        },
        "ar": {
            "action":  "خذ الإلكترونيات إلى مركز تجميع معتمد. لا تلقِها في القمامة أبدًا.",
            "tip":     "مليون هاتف محمول يحتوي على 24 كجم ذهب، 250 كجم فضة، 9000 كجم نحاس.",
            "insight": "النفايات الإلكترونية هي أسرع تيار نفايات نموًا عالميًا.",
            "egypt":   "برنامج النفايات الإلكترونية الوطني (NEEP) يحتاج إلى مشاركة المواطنين.",
        },
    },
    "Hazardous": {
        "en": {
            "action":  "Take to a hazardous waste facility. NEVER pour down the drain or bin.",
            "tip":     "Household hazardous waste includes batteries, paint, pesticides, and cleaning agents.",
            "insight": "Improper disposal contaminates groundwater, affecting millions of people.",
            "egypt":   "The Nile delta's groundwater is critically vulnerable to hazardous waste leaching from informal dumpsites.",
        },
        "ar": {
            "action":  "خذ إلى منشأة النفايات الخطرة. لا تُلقِها في المجاري أو القمامة أبدًا.",
            "tip":     "النفايات المنزلية الخطرة تشمل البطاريات والطلاء والمبيدات.",
            "insight": "التخلص غير السليم يلوث المياه الجوفية ويؤثر على ملايين الأشخاص.",
            "egypt":   "دلتا النيل معرضة بشدة لتسرب النفايات الخطرة من مكبات القمامة غير الرسمية.",
        },
    },
}


def get_recommendation(label: str, lang: str = "en") -> str:
    entry = _KNOWLEDGE.get(label, {}).get(lang, {})
    return entry.get("action", "Dispose responsibly and check local guidelines.")


def get_educational_insight(label: str, lang: str = "en") -> str:
    entry = _KNOWLEDGE.get(label, {}).get(lang, {})
    tip     = entry.get("tip", "")
    insight = entry.get("insight", "")
    egypt   = entry.get("egypt", "")
    return f"💡 {tip}  |  🌍 {egypt}  |  📖 {insight}"


def get_co2_savings(label: str) -> float:
    return CO2_SAVINGS_KG.get(label, 0.0)


def get_sdg_tags(label: str) -> list[str]:
    return SDG_TAGS.get(label, [])


def get_bin_colour(label: str) -> str:
    return RECYCLING_BINS.get(label, "grey")


def get_arabic_label(label: str) -> str:
    return ARABIC_LABELS.get(label, label)
