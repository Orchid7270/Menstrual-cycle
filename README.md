# Menstrual-cycle
'Sundar tar pinakadharahar gangadhar gajacharamamvardhar namo namo'
'Chandra chuda siva sankara parvati ramana ninage namo namo'.Kind of love this song.
Linear regression model for predicting menstrual cycle date

My first git project based on menstrual cycle.Lot of women claims that they do bleed on newmoon or full moon .i have added moon distance(the phases) to the project so that i could try to figure out a relation.But couldnt find any but it doesn't mean that there isn't a relation.Below are my few assumptions and observations.Science is about exploring new things
*Assumptions and Observations*
*During fullmoon the intensity of moon light is higher and during newmoon the intensity is lower.So fewwomen might be exposed to more moonlight during day or night or the amount of water they drink throught the month.Moon does affects tides.During new moon and fullmoon  phases the gravitational pull of the moon and sun on the earth's ocean is aligned.which causes higher high tides and lowe low tides.
*Found a relation between quality of sleep and different symptoms,Diet
*Nature is supposed to be perfect nobody is supposed to get hurt during periods>it could be the lifestyles
*Do we need moonlight similar to sunlight(Vitamin D)?Try to add a column exposure to moon light in minutes
*Amount of water drank during periods sincethe exact percentage of blood in menstrual fluid can vary from woman to woman and even from cycle to cycle. Menstrual blood is a mixture of blood, tissue, and other fluids. The water content in menstrual blood can vary, but it's estimated that:

- Menstrual blood is approximately 50-60% water.

This water content comes from various sources, including:

1. Blood plasma: The liquid portion of blood, which is mostly water.
2. Tissue fluid: Fluid from the uterine lining and other tissues that's shed during menstruation.

The remaining 40-50% of menstrual blood is composed of:

1. Red blood cells
2. White blood cells
3. Platelets
4. Uterine tissue
5. Other cellular debris
Keep in mind that these percentages can fluctuate depending on individual factors, such as hormonal changes, menstrual flow, and overall health.
Well the regression model does output in decimals which is pretty bad for date >Since im not sure if i need to round of the date will think of that later or someone could help.But the error is low.
*Everything that flows has a flow rate Q=AV.Sometimes larger area can reduce the pressure if there is no continouos passage,depends on the velocity.Small area and high velocity does reduce pressure since there is a continuous passage.
*Requirements:
.Skyfield library for moon ditance or the diffrence phase of the moon
pip install skyfield
