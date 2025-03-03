food_recommendations_dict = {
    "Acne": {
        "recommendation": "Consume a balanced diet rich in antioxidants, and avoid greasy foods and dairy.",
        "food_category": [
            {"name": "Antioxidant-rich foods", "icon": "🍓", "image": "/static/images/antioxidants.jpg", 
             "description": "Antioxidants reduce inflammation and help prevent acne. Berries, green tea, and dark chocolate are great sources."},
            {"name": "Leafy greens", "icon": "🥬", "image": "/static/images/leafy_greens.jpg", 
             "description": "Spinach and kale contain vitamins A and C, essential for healthy skin."},
            {"name": "Omega-3 sources", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3s control oil production and reduce redness. Found in salmon, flaxseeds, and walnuts."}
        ],
        "precautions": [
            {"text": "Avoid oily and fried foods.", "icon": "🚫"},
            {"text": "Limit dairy products like cheese and milk.", "icon": "🥛"},
            {"text": "Drink plenty of water to keep skin hydrated.", "icon": "💧"}
        ]
    },
    "Actinic Carcinoma": {
        "recommendation": "Consume foods rich in antioxidants, omega-3 fatty acids, and vitamins C and E to support skin health.",
        "food_category": [
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Boosts collagen production and protects skin from UV damage. Found in citrus fruits, bell peppers, and strawberries."},
            {"name": "Vitamin E sources", "icon": "🥑", "image": "/static/images/vitamin_e.jpg", 
             "description": "Antioxidant that helps prevent skin damage. Found in avocados, almonds, and sunflower seeds."},
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Reduces inflammation and protects against UV damage. Found in salmon, chia seeds, and walnuts."}
        ],
        "precautions": [
            {"text": "Avoid excessive sun exposure.", "icon": "☀️"},
            {"text": "Limit processed and high-fat foods.", "icon": "🍔"},
            {"text": "Drink plenty of water and stay hydrated.", "icon": "💧"}
        ]
    },
    "Atopic Dermatitis": {
        "recommendation": "Eat anti-inflammatory foods that support skin health and help reduce flare-ups.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Berries, leafy greens, and fatty fish help reduce inflammation and soothe skin."},
            {"name": "Healthy fats", "icon": "🥑", "image": "/static/images/healthy_fats.jpg", 
             "description": "Avocados, nuts, and seeds contain healthy fats that help maintain skin barrier function."},
            {"name": "Probiotics", "icon": "🥛", "image": "/static/images/probiotics.jpg", 
             "description": "Fermented foods like yogurt, kefir, and kimchi support gut health, which is linked to skin health."}
        ],
        "precautions": [
            {"text": "Avoid spicy foods and alcohol, which can trigger flare-ups.", "icon": "🌶️"},
            {"text": "Limit dairy and gluten intake if sensitive.", "icon": "🧀"},
            {"text": "Stay hydrated by drinking plenty of water.", "icon": "💧"}
        ]
    },
    "Cellulitis": {
        "recommendation": "Consume foods that support the immune system to help fight infection.",
        "food_category": [
            {"name": "Vitamin C sources", "icon": "🍓", "image": "/static/images/vitamin_c.jpg", 
             "description": "Vitamin C supports immune function and helps heal wounds. Found in citrus fruits, strawberries, and bell peppers."},
            {"name": "Lean proteins", "icon": "🍗", "image": "/static/images/lean_protein.jpg", 
             "description": "Chicken, turkey, and beans provide essential proteins needed for tissue repair."},
            {"name": "Garlic", "icon": "🧄", "image": "/static/images/garlic.jpg", 
             "description": "Garlic has antimicrobial properties that help fight infections."}
        ],
        "precautions": [
            {"text": "Avoid sugary foods that can weaken the immune system.", "icon": "🍬"},
            {"text": "Reduce processed foods and salt intake.", "icon": "🍟"},
            {"text": "Stay hydrated to support your body's healing process.", "icon": "💧"}
        ]
    },
    "Eczema": {
        "recommendation": "Consume foods that reduce inflammation and support skin health.",
        "food_category": [
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3 fatty acids help reduce skin inflammation and are found in salmon, flaxseeds, and walnuts."},
            {"name": "Probiotics", "icon": "🥛", "image": "/static/images/probiotics.jpg", 
             "description": "Fermented foods like yogurt, kefir, and kimchi help support the gut, which plays a role in skin health."},
            {"name": "Vitamin E sources", "icon": "🥑", "image": "/static/images/vitamin_e.jpg", 
             "description": "Vitamin E helps protect the skin from oxidative damage. Found in nuts, seeds, and avocados."}
        ],
        "precautions": [
            {"text": "Avoid foods that trigger flare-ups, like dairy or gluten.", "icon": "🥛"},
            {"text": "Minimize consumption of processed and fried foods.", "icon": "🍟"},
            {"text": "Drink plenty of water to keep skin hydrated.", "icon": "💧"}
        ]
    },
    "Drug Eruptions": {
        "recommendation": "Consume foods that help detoxify the body and support immune health.",
        "food_category": [
            {"name": "Antioxidant-rich foods", "icon": "🍇", "image": "/static/images/antioxidants.jpg", 
             "description": "Antioxidants help neutralize toxins. Include berries, dark chocolate, and leafy greens."},
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Vitamin C helps to support the immune system and repair tissues. Found in citrus fruits and bell peppers."},
            {"name": "Fiber-rich foods", "icon": "🌾", "image": "/static/images/fiber.jpg", 
             "description": "Fiber helps flush out toxins from the body and supports digestion. Found in oats, beans, and vegetables."}
        ],
        "precautions": [
            {"text": "Avoid alcohol and high-sugar foods, which can stress the immune system.", "icon": "🍷"},
            {"text": "Minimize consumption of processed foods and refined sugars.", "icon": "🍩"},
            {"text": "Drink plenty of water to flush out toxins.", "icon": "💧"}
        ]
    },
    "Herpes HPV": {
        "recommendation": "Eat immune-boosting foods and avoid triggers like acidic foods.",
        "food_category": [
            {"name": "Lysine-rich foods", "icon": "🍗", "image": "/static/images/lysine.jpg", 
             "description": "Lysine is an amino acid that helps prevent herpes outbreaks. Found in chicken, turkey, and tofu."},
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Vitamin C helps strengthen the immune system. Found in oranges, strawberries, and broccoli."},
            {"name": "Zinc-rich foods", "icon": "🥩", "image": "/static/images/zinc.jpg", 
             "description": "Zinc helps the immune system function properly. Found in lean meats, pumpkin seeds, and legumes."}
        ],
        "precautions": [
            {"text": "Avoid acidic foods like tomatoes and citrus during outbreaks.", "icon": "🍅"},
            {"text": "Reduce alcohol and tobacco consumption, which can trigger outbreaks.", "icon": "🍷"},
            {"text": "Stay hydrated to support immune function.", "icon": "💧"}
        ]
    },
    "Light Diseases": {
        "recommendation": "Eat foods that are rich in vitamins and antioxidants to protect skin from UV damage.",
        "food_category": [
            {"name": "Carotenoid-rich foods", "icon": "🥕", "image": "/static/images/carrots.jpg", 
             "description": "Carotenoids like beta-carotene help protect skin from UV damage. Found in carrots, sweet potatoes, and spinach."},
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3s reduce inflammation and protect the skin. Found in salmon, chia seeds, and flaxseeds."},
            {"name": "Vitamin E sources", "icon": "🥑", "image": "/static/images/vitamin_e.jpg", 
             "description": "Vitamin E helps prevent skin damage from sun exposure. Found in avocados, nuts, and sunflower seeds."}
        ],
        "precautions": [
            {"text": "Avoid excessive sun exposure and wear sunscreen.", "icon": "☀️"},
            {"text": "Limit alcohol consumption as it can damage the skin.", "icon": "🍷"},
            {"text": "Stay hydrated by drinking water and eating hydrating foods.", "icon": "💧"}
        ]
    },
    "Lupus": {
        "recommendation": "Consume anti-inflammatory foods and avoid foods that can trigger flare-ups.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Berries, leafy greens, and fatty fish help reduce inflammation and soothe skin."},
            {"name": "Vitamin D-rich foods", "icon": "🍄", "image": "/static/images/vitamin_d.jpg", 
             "description": "Vitamin D supports immune health and can help reduce lupus flare-ups. Found in fortified foods and mushrooms."},
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3s help reduce inflammation. Found in salmon, walnuts, and flaxseeds."}
        ],
        "precautions": [
            {"text": "Avoid processed foods and sugary snacks.", "icon": "🍩"},
            {"text": "Minimize sodium intake to manage blood pressure.", "icon": "🧂"},
            {"text": "Stay hydrated to maintain overall health.", "icon": "💧"}
        ]
    },
    "Melanoma": {
        "recommendation": "Consume foods rich in vitamins C and E, and omega-3 fatty acids to support skin health.",
        "food_category": [
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Boosts collagen production and protects against sun damage. Citrus fruits, bell peppers, and strawberries are great choices."},
            {"name": "Vitamin E sources", "icon": "🥑", "image": "/static/images/vitamin_e.jpg", 
             "description": "Antioxidant that helps prevent skin damage. Found in avocados, almonds, and sunflower seeds."},
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Reduces inflammation and protects against UV damage. Found in salmon, chia seeds, and walnuts."}
        ],
        "precautions": [
            {"text": "Avoid excessive alcohol consumption.", "icon": "🍷"},
            {"text": "Reduce red meat intake to lower inflammation.", "icon": "🥩"},
            {"text": "Use sunscreen and eat hydrating foods like cucumbers.", "icon": "☀️"}
        ]
    },
    "Poison Ivy": {
        "recommendation": "Eat foods that reduce inflammation and support immune function to aid healing.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Blueberries, spinach, and fatty fish help reduce inflammation and promote healing."},
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Vitamin C supports immune function and helps heal skin. Found in citrus fruits and bell peppers."},
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3s help reduce inflammation. Found in salmon, chia seeds, and flaxseeds."}
        ],
        "precautions": [
            {"text": "Avoid scratching or touching affected areas.", "icon": "🚫"},
            {"text": "Limit alcohol consumption, which may worsen inflammation.", "icon": "🍷"},
            {"text": "Stay hydrated to help flush out toxins.", "icon": "💧"}
        ]
    },
    "Psoriasis": {
        "recommendation": "Consume anti-inflammatory foods that help control flare-ups and reduce skin irritation.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Berries, green tea, and leafy greens can help reduce inflammation and improve skin health."},
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3s help reduce skin inflammation. Found in salmon, flaxseeds, and walnuts."},
            {"name": "Vitamin D sources", "icon": "🍄", "image": "/static/images/vitamin_d.jpg", 
             "description": "Vitamin D is vital for skin health and immune function. Found in fortified foods and mushrooms."}
        ],
        "precautions": [
            {"text": "Avoid alcohol and tobacco, which can worsen symptoms.", "icon": "🍷"},
            {"text": "Limit processed and fried foods, which can trigger inflammation.", "icon": "🍟"},
            {"text": "Drink plenty of water to keep skin hydrated.", "icon": "💧"}
        ]
    },
    "Benign Tumors": {
        "recommendation": "Consume foods that help reduce inflammation and support overall health.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Berries, leafy greens, and omega-3 rich foods like salmon help reduce inflammation."},
            {"name": "Fiber-rich foods", "icon": "🌾", "image": "/static/images/fiber.jpg", 
             "description": "Fiber helps support the digestive system and reduce the risk of tumor growth. Found in whole grains, legumes, and vegetables."},
            {"name": "Cruciferous vegetables", "icon": "🥦", "image": "/static/images/cruciferous_vegetables.jpg", 
             "description": "Broccoli, cauliflower, and kale contain compounds that may help prevent cancer development."}
        ],
        "precautions": [
            {"text": "Avoid excessive intake of red meat and processed meats.", "icon": "🥩"},
            {"text": "Minimize alcohol consumption, which may increase cancer risk.", "icon": "🍷"},
            {"text": "Stay hydrated with water to support overall health.", "icon": "💧"}
        ]
    },
    "Systemic Disease": {
        "recommendation": "Consume foods that strengthen the immune system and support organ health.",
        "food_category": [
            {"name": "Antioxidant-rich foods", "icon": "🍓", "image": "/static/images/antioxidants.jpg", 
             "description": "Antioxidants, found in berries and green tea, support the immune system and reduce oxidative stress."},
            {"name": "Healthy fats", "icon": "🥑", "image": "/static/images/healthy_fats.jpg", 
             "description": "Healthy fats, like those in avocados and nuts, are crucial for reducing inflammation and promoting overall health."},
            {"name": "Lean proteins", "icon": "🍗", "image": "/static/images/lean_protein.jpg", 
             "description": "Protein supports tissue repair and immune function. Found in chicken, fish, and beans."}
        ],
        "precautions": [
            {"text": "Avoid excessive intake of sugar and refined carbs.", "icon": "🍩"},
            {"text": "Minimize processed and fast foods that can impair immune function.", "icon": "🍟"},
            {"text": "Drink plenty of water and maintain hydration.", "icon": "💧"}
        ]
    },
    "Ringworm": {
        "recommendation": "Eat foods that help fight infections and strengthen the immune system.",
        "food_category": [
            {"name": "Garlic", "icon": "🧄", "image": "/static/images/garlic.jpg", 
             "description": "Garlic has antifungal properties that help fight infections like ringworm."},
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Vitamin C boosts the immune system. Found in citrus fruits, bell peppers, and broccoli."},
            {"name": "Probiotics", "icon": "🥛", "image": "/static/images/probiotics.jpg", 
             "description": "Fermented foods like yogurt and kefir support gut health, which is linked to overall immunity."}
        ],
        "precautions": [
            {"text": "Avoid scratching infected areas to prevent spreading.", "icon": "🚫"},
            {"text": "Limit alcohol consumption to maintain immune function.", "icon": "🍷"},
            {"text": "Drink plenty of water to stay hydrated and flush toxins.", "icon": "💧"}
        ]
    },
    "Urticarial Hives": {
        "recommendation": "Consume foods that help reduce inflammation and support skin healing.",
        "food_category": [
            {"name": "Omega-3 fatty acids", "icon": "🐟", "image": "/static/images/omega3.jpg", 
             "description": "Omega-3 fatty acids help control inflammation. Found in salmon, flaxseeds, and walnuts."},
            {"name": "Antioxidant-rich foods", "icon": "🍇", "image": "/static/images/antioxidants.jpg", 
             "description": "Antioxidants reduce inflammation and promote skin healing. Found in berries, green tea, and dark chocolate."},
            {"name": "Vitamin E sources", "icon": "🥑", "image": "/static/images/vitamin_e.jpg", 
             "description": "Vitamin E protects the skin from oxidative stress. Found in almonds, sunflower seeds, and avocados."}
        ],
        "precautions": [
            {"text": "Avoid spicy foods and alcohol, which can trigger hives.", "icon": "🌶️"},
            {"text": "Limit foods that contain high levels of histamine, like aged cheeses.", "icon": "🧀"},
            {"text": "Stay hydrated with plenty of water.", "icon": "💧"}
        ]
    },
    "Vascular Tumors": {
        "recommendation": "Consume foods that reduce inflammation and support overall health.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Berries, leafy greens, and fatty fish help reduce inflammation."},
            {"name": "Fiber-rich foods", "icon": "🌾", "image": "/static/images/fiber.jpg", 
             "description": "Fiber supports the digestive system and may help lower the risk of tumor growth."},
            {"name": "Cruciferous vegetables", "icon": "🥦", "image": "/static/images/cruciferous_vegetables.jpg", 
             "description": "Broccoli, kale, and cauliflower may help reduce cancer risk."}
        ],
        "precautions": [
            {"text": "Avoid excessive red meat and processed meats.", "icon": "🥩"},
            {"text": "Minimize alcohol consumption, which may increase cancer risk.", "icon": "🍷"},
            {"text": "Drink plenty of water to maintain hydration.", "icon": "💧"}
        ]
    },
    "Vasculitis": {
        "recommendation": "Consume anti-inflammatory foods and foods that support immune function.",
        "food_category": [
            {"name": "Anti-inflammatory foods", "icon": "🫐", "image": "/static/images/antiinflammatory.jpg", 
             "description": "Berries, green tea, and omega-3 rich foods help reduce inflammation."},
            {"name": "Lean proteins", "icon": "🍗", "image": "/static/images/lean_protein.jpg", 
             "description": "Lean proteins help tissue repair. Found in chicken, turkey, and legumes."},
            {"name": "Fiber-rich foods", "icon": "🌾", "image": "/static/images/fiber.jpg", 
             "description": "Fiber helps reduce inflammation. Found in oats, beans, and vegetables."}
        ],
        "precautions": [
            {"text": "Avoid processed and high-sodium foods.", "icon": "🍔"},
            {"text": "Minimize alcohol consumption.", "icon": "🍷"},
            {"text": "Stay hydrated with water to help reduce inflammation.", "icon": "💧"}
        ]
    },
    "Viral Infections": {
        "recommendation": "Consume immune-boosting foods to help fight infections and support recovery.",
        "food_category": [
            {"name": "Garlic", "icon": "🧄", "image": "/static/images/garlic.jpg", 
             "description": "Garlic has antimicrobial properties that can help fight infections."},
            {"name": "Vitamin C sources", "icon": "🍊", "image": "/static/images/vitamin_c.jpg", 
             "description": "Vitamin C boosts the immune system. Found in citrus fruits and bell peppers."},
            {"name": "Probiotics", "icon": "🥛", "image": "/static/images/probiotics.jpg", 
             "description": "Fermented foods like yogurt, kefir, and kimchi help support the gut and immune system."}
        ],
        "precautions": [
            {"text": "Avoid alcohol, which can weaken the immune system.", "icon": "🍷"},
            {"text": "Limit processed foods that may impair immune function.", "icon": "🍔"},
            {"text": "Drink plenty of fluids to stay hydrated.", "icon": "💧"}
        ]
    }
}
