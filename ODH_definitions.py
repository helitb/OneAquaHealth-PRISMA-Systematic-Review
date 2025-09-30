# ODH_definitions.py
# Here we define everything that has to do with ODH and OAH 


OneAquaHealth = f"""
Urban aquatic ecosystems are extremely relevant connectors between people, animals, and plants, making cities more biodiverse and sustainable. 
Yet, these ecosystems are often confronted with lack of space, vegetation cuts, artificialization, and other urbanisation processes. 
This degradation can lead to numerous disservices to humans regarding emerging pathogens, decreasing disease resistance, climate change impacts and other health concerns in cities.
OneAquaHealth is a reasearch project. OneAquaHealth aims to improve the sustainability and integrity of freshwater ecosystems in urban environments.  
The project will identify early warning indicators and enhance environmental monitoring by investigating the interconnection of ecosystem health and human wellbeing. 
"""

OneDigitalHealth = f"""
One Digital Health (ODH) is a proposed unified structure. 
The conceptual framework of the One Digital Health Steering Wheel is built around two keys (ie, One Health and digital health), three perspectives (ie, individual health and well-being, population and society, and ecosystem), and five dimensions (ie, citizens' engagement, education, environment, human and veterinary health care, and Healthcare Industry 4.0). 
One Digital Health aims to digitally transform future health ecosystems, by implementing a systemic health and life sciences approach that takes into account broad digital technology perspectives on human health, animal health, and the management of the surrounding environment.
"""

OneAquaHealth_indicators = f"""Ecological and Human Health Indicators:
Key characteristics include sensitivity, reliability, relevance, practicality, and validity. 
Ecological indicators measure ecosystem health and the impact of human activities, while human health indicators assess population well-being and public health effectiveness.
Examples for Indicators:
Ecological: Biological (e.g., species diversity), Physical (e.g., water quality), Chemical (e.g., pollutant levels).
Human Health: Mortality rates, life expectancy, disease prevalence, and quality of life measures.
"""

ODH_UT_definitions = {
	'ODHdef_ODH': "One Digital Health (ODH) is a proposed unified structure. The conceptual framework of the One Digital Health Steering Wheel is built around two keys (ie, One Health and digital health), three perspectives (ie, individual health and well-being, population and society, and ecosystem), and five dimensions (ie, citizens' engagement, education, environment, human and veterinary health care, and Healthcare Industry 4.0). One Digital Health aims to digitally transform future health ecosystems, by implementing a systemic health and life sciences approach that takes into account broad digital technology perspectives on human health, animal health, and the management of the surrounding environment.",
	'ODHdef_onehealth': "'One Health' definition is 'One Health is an integrated and unifying approach to sustainably balancing and optimizing the linked and interdependent health of people, animals, and ecosystems.'",
	'ODHdef_digitalhealth': "'Digital Health'  definition is 'Digital Health is the use of technology in medicine and other health professions to manage illnesses and health risks and promote wellness.'",
	'ODHdef_individualhealth': "'Individual Health and Well-being' definition is 'Individual Health and Well-being rely on an individual's physical, mental, and social well-being, focusing on personal health promotion and disease prevention.'",
	'ODHdef_population': "Population and Society' definition is 'The health of groups of people and animals, considering factors like social determinants and community-level interventions.'",
	'ODHdef_ecosystems': "'Ecosystem' definition is 'The health of the broader environment, including the interplay of living organisms and their surroundings, influences human and animal health.'",
	'ODHdef_citizens': "'Citizen’s Engagement' definition is 'The involvement of individuals in health-related decision-making processes fosters collaboration between healthcare providers and the public.'",
	'ODHdef_education': "'Education' definition is 'Education promotes awareness, understanding, and acquisition of information and knowledge, empowering individuals, and communities to make informed decisions.'",
	'ODHdef_envmoni': "'Environment monitoring' definition is 'Surveillance of the external surroundings that impact health, including biological, chemical, physical, social, and cultural elements.'",
	'ODHdef_HVH': "'Human and veterinary healthcare' definition is 'Human and veterinary healthcare rely on all aspects of providing and distributing health services to a patient population (human or animal).'",
	'ODHsynonyms_I4': "'Healthcare Industry (4.0)' definition is 'The Healthcare Industry (4.0) integrates advanced technologies like artificial intelligence, big data, and IoT into healthcare systems to improve efficiency and patient outcomes.'",
	'ODHsynonyms_onehealth': "'One Health' synonyms should be 'Global Health', 'Unified Health', 'Integrated Health'",
	'ODHsynonyms_digitalhealth': "'Digital Health' synonyms should be 'eHealth', 'Health Tech', 'Health Informatics', 'Medical Informatics', 'Nursing Informatics'",
	'ODHsynonyms_individualhealth': "'Individual Health and Well-being' synonyms should be 'Personal Health', 'Wellness'",
	'ODHsynonyms_population': "Population and Society' synonyms should be 'Community Health', 'Public Health'",
	'ODHsynonyms_ecosystems': "'Ecosystem' synonyms should be 'Environmental Health', 'Ecological Health'",
	'ODHsynonyms_citizens': "'Citizen’s Engagement' synonyms should be 'Public Participation', 'Community Involvement'",
	'ODHsynonyms_education': "'Education' synonyms should be 'Literacy', 'Knowledge', 'Learning', 'Dissemination'",
	'ODHsynonyms_envmoni': "'Environment monitoring' synonyms should be 'Environmental Factors', 'Surroundings monitoring', 'Ecological indicators tracking'",
	'ODHsynonyms_HVH': "'Human and veterinary healthcare' synonyms should be 'Delivery of health care', 'Medical Care', 'Animal Health Services'",
	'ODHsynonyms_I4': "'Healthcare Industry (4.0)' synonyms should be 'Healthcare Industry', 'Digital Transformation in Health'"
},

ODH_SW_definitions = {
	'ODHSWmeaning': "ODH-SW means 'One Digital Health - Steering Wheel'. ODH-SW is the formal representation of the One Digital Health framework.",
	'ODHSWk': "ODH-SW is the generic model defining the integration of 'One Health' and 'Digital Health' as key concepts.",
	'ODHSWp': "ODH-SW is the generic model defining the integration of 'Individual Health and well-being', 'Population and Society', and 'Ecosystem' as perspective concepts.", 
	'ODHSWp': "ODH-SW is the generic model defining the integration of 'Education', 'Environment', 'Human and Veterinary Healthcare', 'Healthcare Industry (4.0)', and 'Citizen’s engagement' as dimension terms and concepts.",
	'ODHSWkpd_comp': "ODH-SW components are 'One Health', 'Digital Health', Individual Health and well-being', 'Population and Society', 'Ecosystem', 'Education', 'Environment', 'Human and Veterinary Healthcare', 'Healthcare Industry (4.0)', 'Citizen’s engagement'.",
	'ODHSWkpd_comb1': "ODH-SW all keys, perspectives, and dimensions can be mutually combined.",
	'ODHSWkpd_comb2': "ODH-SW all components can be mutually combined.",
	'ODHSWlayers': "ODH-SW keys, perspectives, and dimensions  are layer.",
	'ODHSWkpd': "The terms defining the ODH-SW keys, perspectives, and dimensions may be adjusted to be more specific (as narrower terms) or more general (as broader terms), depending on the context and application use cases."
}


OAH_definitions = {
	'Key characteristics of ecological and human health indicators':  "Key characteristics of ecological and human health indicators are Sensitivity (Response to changes in environmental conditions), Reliability (Provide consistent and accurate information), Relevance (Directly related to the ecosystem being studied), Practicality (Easily measured and monitored), Validity (Measures what it is intended to measure).",
	'Ecological Indicator': "An ecological indicator is a measure of the health and condition of an ecosystem. It provides information about environmental changes and can be used to assess the impact of human activities on the natural world.",
	'Human health indicator': "Human health indicators is a measure of the overall health and well-being of a population or individual. It provides insights into health status, trends, and disparities. These indicators can be used to assess the effectiveness of public health interventions, monitor disease burden, and identify populations at risk.",
	'Kinds of ecological indicators': "Kinds of ecological indicators are Biological indicators (eg, Species diversity, population numbers, and presence of indicator species -like amphibians or lichens-), Physical indicators (eg, water quality, air quality, soil erosion), and climate patterns, Chemical indicators (eg, Nutrient levels, pollutant concentrations, and pH levels).",
	'Kinds of human health indicators': " Mortality rates (Measures the number of deaths within a population), Life expectancy (Average lifespan of a population), Infant mortality rate (Death rate of infants under one year old), Disease prevalence (Number of cases of a specific disease in a population), Disease incidence (Number of new cases of a specific disease in a population during a specific period of time), BMI (Body Mass Index), Health-related quality of life (Overall well-being and satisfaction with health).",

}
