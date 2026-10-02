"""
Hand-curated lexicons used for
  1. word-level language tagging (which word is English / Hindi / Gujarati),
  2. a rule-based "second opinion" that is blended with the ML model when the
     model is unsure (very short inputs like "kem cho").

Romanized Hindi and Gujarati have no fixed spelling ("che", "chhe", "6e"),
so the most common spelling variants are listed explicitly.
"""

# --- Romanized Gujarati (Gujlish) high-precision marker words ---------------
GUJARATI_ROMAN = set("""
che chhe chhu chu cho chho chhie chie chiye che? hatu hati hata hatun hase hashe
kem kemcho shu su shun sha nathi nthi nathy nathe nai nahi_gu
maja majama majja majjama maza mazama saru saras saaru saras jordar mast_gu
tame tamne tamaru tamari tamara tamaro tamej ame amne amaru amari amara aapde aapdu aapdi
mane maru mari mara maro mare tane taru tari tara taro tare ene eno eni ena aene aena tene
kya_gu kyare kyathi kyan tya tyare ahi ahiya ahiyan tyaa jyare jyan jya
ketla ketlu ketli ketlo ketli kevi kevu kevo keva kon koni kone konu kai kaik kaink
jau jaiye jaie jaishu jaish jase jashe javu javanu javano jajo jaje gayo gayu gai gaya
aavu aavje aavjo aavso aavsho aavish aavyo aavya aave aavvu aavde aavdtu aavdatu
karu karo kare karvu karvanu karje karjo karish karshe karsho kariye karie kari karyu karyo
lidhu lidho lidhi dejo daish dish aapje aapjo
ghanu ghani ghana bahu bau khub khubaj ekdum_gu
haju hamna atyare pachi pachhi pachu aagal aagad saathe sathe
bhukh lagi lagyu lage lagse lagshe pade padse padshe padto
jamvanu jamya jami jamishu jamva nasto chaa
ungh unghvu thaki thakyo thaki_gu
varsad thandi garmi_gu dhire dhimu jaldi_gu moduu modu
kharekhar sachu khotu
malie maliye malishu malvanu mali malyu
kahish kahevu kehvu kahi kaho kehje kahje bolyo bolo_gu
rehje rahu raho rehva rahish
samjanu samjhayu samjavo fari
nu ni na_gu no ne_gu thi ma_gu par_gu
pan_gu toh_gu
pahochu pahochi pahochine pahochyo
utari utarvanu ubho
jou joyu joyo jovu jo_gu
jitshe ramva ramvani ramishu
chuki chukyo
monghu sastu ochu ochho bhav
mummy_gu dadi_gu ben bena
shubhechha shubhkamna
padharo aavo_gu
bagdi bagdyu
vandho vaandho
chinta_gu
saachvine sachvi dhyan_gu
avaj
tabiyat dawakhane batavu
pivi piso peevi
hasavi hasvu
gussa_gu naraz_gu
navu navi
mithai_gu lagan lagna sagai
aaje kale
""".split())

# --- Romanized Hindi (Hinglish) high-precision marker words -----------------
HINDI_ROMAN = set("""
hai hain hu hoon hun ho_hi tha thi the_hi thay
kya kyaa kyun kyu kyon kaise kese kaisa kaisi kaun kahan kaha kab kitna kitne kitni
nahi nahin nhi nai_hi mat
mai main mein me_hi mujhe mujhse mera meri mere tum tumhe tumhara tumhari tera teri tere tujhe tujhse
aap aapka aapki aapke aapse hum humein hamara hamari usko usse uska uski unko unka unki wo woh ye yeh
raha rahi rahe rha rhi rhe rahna
gaya gayi gaye gya gyi
karna karta karti karte karunga karungi karenge kar_hi kiya kiye kijiye karo_hi
jaana jana jaata jaati jaate jayega jayegi jaunga
aana aata aati aate aaunga aaungi aayega aaya aayi aaoge
bahut bohot bahot accha acha achha achhi badhiya zabardast theek thik
abhi kabhi sabhi bhi phir fir baad pehle
yaar yr bhaiya didi_hi
chal chalo chalte nikalte
milte milenge mila mile
pata bata batao batana batata bolo_hi bola bol
khana khaya khaane khane piyoge peeni
neend sona_hi
thand sardi garmi_hi baarish
dekha dekho dekhna
samjhe samajh samjhao
dheere jaldi_hi
sach jhooth yakeen
lagi_hi lag laga
paise rupaye mehenga sasta
ghar_hi gaadi_hi
liya lena dena dunga dungi
saath sath
toh to_hi
wala wali wale
kuch koi sab
gussa_hi naraz_hi
shaadi sagai_hi
hoga hogi honge
ruk ruko
dard
""".split())

# Words that occur in BOTH Hindi and Gujarati romanized text (not decisive)
SHARED_INDIC_ROMAN = set("""
bhai bhagwan jai shree shri ram krishna radhe mummy papa dadi mama didi chai
paisa pagal gaando mast ekdum jaldi garmi kal aaj ghar gaadi chinta tension
haan ha na nai ne ma par pan to dhyan gussa naraz mithai
""".split())

# Strip the disambiguation suffixes used above (e.g. "nai_hi" -> "nai")
def _clean(words):
    return {w.split("_")[0] for w in words}

GUJARATI_ROMAN = _clean(GUJARATI_ROMAN)
HINDI_ROMAN = _clean(HINDI_ROMAN)
SHARED_INDIC_ROMAN = _clean(SHARED_INDIC_ROMAN)

# Remove words that ended up in both lists from the decisive sets
_overlap = GUJARATI_ROMAN & HINDI_ROMAN
GUJARATI_ROMAN -= _overlap
HINDI_ROMAN -= _overlap
SHARED_INDIC_ROMAN |= _overlap

# --- Common English words ----------------------------------------------------
ENGLISH = set("""
a about above after again against all also always am an and any are aren't as at
be because been before being below between both but by can can't cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further had hadn't
has hasn't have haven't having he her here hers herself him himself his how i if in into
is isn't it it's its itself just let's me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should shouldn't so some such
than that that's the their theirs them themselves then there there's these they this those
through to too under until up very was wasn't we were weren't what when where which while
who whom why will with won't would wouldn't you your yours yourself yourselves
hello hi hey thanks thank please sorry yes yeah ok okay sure good great nice awesome cool
bad best better love like want need know think see look come go get make take give tell
say said call work home office meeting deadline project exam college school class student
teacher friend family mom dad brother sister bro sir madam today tomorrow yesterday tonight
morning evening night time day week month year minute minutes hour hours late early
again really very much many little big small new old happy sad tired hungry sleep food
water tea coffee lunch dinner breakfast party movie music song video photo phone mobile
battery charger internet wifi app message whatsapp email send sent reply read write
open close closed start stop finish done ready free busy money price cost cheap expensive
shop market car bike bus train flight traffic road left right straight here there where
reach reached coming going went gone back way wait waiting fast slow help problem
weather rain raining hot cold outside inside umbrella fan light power working
birthday congratulations welcome beautiful amazing crazy funny kidding truth lie believe
understand explain speak listen quiet name live living business startup learn coding
data model language processing natural artificial intelligence machine learning report
document attached questions share submit assignment assignments policy government
team won hackathon library book garden football cricket match score win play played
planning trip mountains restaurant nearby recommend password account reset budget
discuss rescheduled delayed conference everyone dreams job place click continue terms
conditions apply website evening morning good night sweet chilling nothing going on
what's i'm you're he's she's we're they're i'll you'll we'll i've you've don't
""".split())

# --- Native-script disambiguation for Devanagari ----------------------------
MARATHI_DEVANAGARI = set("आहे आहेत आहेस नाही आणि मी तू काय होते होता होती माझे माझा माझी तुझे तुझा तुम्ही आम्ही आपण करतो करते जातो येतो कसा कशी कुठे झाले खूप".split())
NEPALI_DEVANAGARI = set("छ छन् छु छौ हुन्छ गर्नु मेरो तिमी तपाईं तपाईंलाई हामी थियो भयो गर्छ धेरै राम्रो कस्तो छैन".split())
HINDI_DEVANAGARI = set("है हैं हूँ हूं था थी थे और मैं क्या नहीं का की के को में से पर यह वह मेरा तुम आप हम रहा रही गया बहुत कहाँ कैसे कब क्यों रहते करते जाते अच्छा लेकिन भी नाम ये वो अब हो".split())
