# =============================================================================
# CELL 0: FHRR Research Dataset Schema (v2)
# =============================================================================
fhrr_research_dataset = {
    # -------------------------------------------------------------------------
    # 1. VOCABULARY BASE
    # -------------------------------------------------------------------------
    'vocab': {
        'categories': {
            'benda_padat': ['batu','besi','kayu','kaca','emas','perak','tembok','meja','kursi','pintu','gergaji','benteng','buku','papan'],
            'benda_cair': ['air','minyak','darah','susu','madu','cairan','hujan','sungai'],
            'benda_energi': ['api','listrik','cahaya','panas','dingin','angin','petir','sinar'],
            'hewan': ['anjing','kucing','burung','ikan','ular','gajah','harimau','kupu_kupu','lebah','lumba_lumba'],
            'tumbuhan': ['pohon','rumput','bunga','mangga','padi','kelapa','anggrek','bambu'],
            'manusia': ['orang','anak','ibu','ayah','guru','dokter','teman','musuh','pemimpin','murid','rakyat','pasien'],
            'konsep': ['kebenaran','keadilan','kebebasan','keamanan','bahagia','dukacita','cinta','benci','takut','berani','harapan','putus_asa'],
            'aksi_fisik': ['lari','jalan','lompat','terbang','berenang','duduk','tidur','bangun','makan','minum','tendang','tangkap','lempar','potong','ukir','masak','turun'],
            'aksi_komunikasi': ['katakan','tulis','baca','dengar','lihat','tanya','jawab','jelaskan','peringatkan','puji','kritik','pidato'],
            'aksi_sosial': ['bantu','tipu','lindungi','serang','dukung','khianati','hormati','hina'],
            'ukuran': ['besar','kecil','panjang','pendek','lebar','sempit','tinggi','rendah','tebal','tipis'],
            'warna': ['merah','biru','hijau','kuning','hitam','putih','ungu','oranye','coklat','abu'],
            'kualitas': ['baik','buruk','indah','jelek','sehat','sakit','kuat','lemah','cepat','lambat','mudah','sulit','tajam'],
            'emosi': ['senang','sedih','marah','takut','jijik','terkejut','tenang','gembira','cemas','bangga','malu'],
            'keadaan': ['baru','lama','muda','tua','basah','kering','panas','dingin','bersih','kotor','rapi','berantakan','jernih'],
            'kecepatan': ['cepat','lambat','tiba_tiba','perlahan','konstan','fluktuatif'],
            'waktu': ['sekarang','nanti','kemarin','besok','selalu','pernah','segera','akhirnya','pagi','siang','malam'],
            'ruang': ['sini','sana','atas','bawah','dalam','luar','dekat','jauh','tengah','depan','belakang'],
            'kuantitas': ['banyak','sedikit','semua','sebagian','kosong','penuh','cukup','kurang'],
            'relasi': ['di','ke','dari','dalam','luar','antara','melalui','melewati','sebelum','sesudah','selama','ketika','sampai','sejak','karena','sebab','akibat','maka','jadi','oleh','untuk','dengan','tanpa','bagi','milik','bersama','sendiri','dan','atau','tetapi','namun','meski','walaupun','jika','kecuali']
        },
        'poles': {
            'ukuran': {'positive': ['besar','panjang','lebar','tinggi','tebal'], 'negative': ['kecil','pendek','sempit','rendah','tipis']},
            'warna': {'positive': ['merah','hijau','hitam','ungu','coklat'], 'negative': ['biru','kuning','putih','oranye','abu']},
            'kualitas': {'positive': ['baik','indah','sehat','kuat','cepat','mudah','tajam'], 'negative': ['buruk','jelek','sakit','lemah','lambat','sulit']},
            'emosi': {'positive': ['senang','marah','jijik','tenang','cemas','malu'], 'negative': ['sedih','takut','terkejut','gembira','bangga']},
            'kecepatan': {'positive': ['cepat','tiba_tiba'], 'negative': ['lambat','perlahan']},
            'keadaan': {'positive': ['baru','muda','basah','panas','bersih','rapi','jernih'], 'negative': ['lama','tua','kering','dingin','kotor','berantakan']}
        }
    },

    # -------------------------------------------------------------------------
    # 2. OBSERVATION CORPUS (Fakta dunia)
    # -------------------------------------------------------------------------
    'observations': [
        {'id':'o1', 'bindings':{'agen':'anak','predikat':'makan','pasien':'mangga','lokasi':'kebun','waktu':'kemarin'}, 'type':'event'},
        {'id':'o2', 'bindings':{'agen':'guru','predikat':'tulis','lokasi':'papan','atribut':'putih','instrumen':'kapur'}, 'type':'event'},
        {'id':'o3', 'bindings':{'agen':'dokter','predikat':'bantu','pasien':'orang','atribut':'sakit','lokasi':'rumah_sakit'}, 'type':'event'},
        {'id':'o4', 'bindings':{'agen':'burung','predikat':'terbang','arah':'atas','lokasi':'pohon','atribut':'tinggi'}, 'type':'event'},
        {'id':'o5', 'bindings':{'agen':'ikan','predikat':'berenang','lokasi':'sungai','atribut':'jernih'}, 'type':'event'},
        {'id':'o6', 'bindings':{'agen':'kucing','predikat':'tidur','lokasi':'bawah','pasien':'meja','atribut':'besar'}, 'type':'event'},
        {'id':'o7', 'bindings':{'agen':'ayah','predikat':'potong','pasien':'kayu','instrumen':'gergaji','atribut':'tajam'}, 'type':'event'},
        {'id':'o8', 'bindings':{'agen':'ibu','predikat':'masak','pasien':'nasi','lokasi':'dapur','atribut':'bersih'}, 'type':'event'},
        {'id':'o9', 'bindings':{'agen':'pemimpin','predikat':'pidato','lokasi':'depan','pasien':'rakyat','atribut':'banyak'}, 'type':'event'},
        {'id':'o10', 'bindings':{'agen':'musuh','predikat':'serang','pasien':'benteng','instrumen':'pasukan','atribut':'kuat'}, 'type':'event'},
    ],

    # -------------------------------------------------------------------------
    # 3. REASONING PATTERNS (Cara menyimpulkan)
    # -------------------------------------------------------------------------
    'reasoning_patterns': [
        {'id':'r1', 'name':'causal_eat_happy', 'premise':{'predikat':'makan'}, 'conclusion':{'atribut':'senang'}, 'mechanism':'transform', 'confidence':0.75, 'explanation':'makan mengakibatkan senang'},
        {'id':'r2', 'name':'causal_rain_wet', 'premise':{'predikat':'turun','agen':'hujan'}, 'conclusion':{'atribut':'basah'}, 'mechanism':'transform', 'confidence':0.8, 'explanation':'hujan membuat basah'},
        {'id':'r3', 'name':'spatial_fly_up', 'premise':{'predikat':'terbang'}, 'conclusion':{'arah':'atas'}, 'mechanism':'transform', 'confidence':0.9, 'explanation':'terbang menuju atas'},
        {'id':'r4', 'name':'antonym_size', 'premise':{'atribut':'besar'}, 'conclusion':{'atribut':'kecil'}, 'mechanism':'transform', 'confidence':1.0, 'explanation':'besar adalah lawan kecil', 'invert':True},
        {'id':'r5', 'name':'role_instrument', 'premise':{'predikat':'potong'}, 'conclusion':{'instrumen':'gergaji'}, 'mechanism':'association', 'confidence':0.6, 'explanation':'potong biasanya pakai gergaji'},
        {'id':'r6', 'name':'entailment_communication', 'premise':{'predikat':'katakan','agen':'guru'}, 'conclusion':{'predikat':'dengar','agen':'murid'}, 'mechanism':'role_swap', 'confidence':0.7, 'explanation':'guru mengatakan murid mendengar'},
    ],

    # -------------------------------------------------------------------------
    # 4. QUESTION-ANSWER PAIRS (Cara menjawab)
    # -------------------------------------------------------------------------
    'qa_pairs': [
        {'id':'qa1', 'question':'siapa yang makan mangga?', 'q_focus':['predikat','pasien'], 'answer_role':'agen', 'source':'o1', 'reasoning':'cari agen dari peristiwa makan mangga'},
        {'id':'qa2', 'question':'apa yang dimakan anak?', 'q_focus':['agen','predikat'], 'answer_role':'pasien', 'source':'o1', 'reasoning':'cari pasien dari aksi makan oleh anak'},
        {'id':'qa3', 'question':'di mana anak makan?', 'q_focus':['agen','predikat'], 'answer_role':'lokasi', 'source':'o1', 'reasoning':'cari lokasi dari peristiwa makan'},
        {'id':'qa4', 'question':'kapan anak makan mangga?', 'q_focus':['agen','predikat','pasien'], 'answer_role':'waktu', 'source':'o1', 'reasoning':'cari waktu dari peristiwa'},
        {'id':'qa5', 'question':'mengapa tanah basah?', 'q_focus':['atribut'], 'answer_role':'agen', 'source':'o2', 'reasoning':'inferensi kausal dari basah ke penyebab'},
        {'id':'qa6', 'question':'apa yang digunakan ayah untuk memotong?', 'q_focus':['agen','predikat'], 'answer_role':'instrumen', 'source':'o7', 'reasoning':'cari instrumen dari aksi potong'},
        {'id':'qa7', 'question':'bagaimana perasaan anak setelah makan?', 'q_focus':['agen','predikat'], 'answer_role':'atribut', 'source':'o1', 'reasoning':'inferensi emosi dari aksi makan'},
    ],

    # -------------------------------------------------------------------------
    # 5. EXPLANATION TEMPLATES (Cara menjelaskan)
    # -------------------------------------------------------------------------
    'explanation_templates': [
        {'id':'e1', 'pattern':'decompose_svo', 'strategy':'uraikan peran agen-predikat-pasien', 'template':'{agen} melakukan {predikat} terhadap {pasien}'},
        {'id':'e2', 'pattern':'decompose_full', 'strategy':'uraikan semua peran', 'template':'{agen} {predikat} {pasien} di {lokasi} pada {waktu} dengan {instrumen}'},
        {'id':'e3', 'pattern':'causal_chain', 'strategy':'jelaskan rantai kausal', 'template':'karena {premise}, maka {conclusion}'},
        {'id':'e4', 'pattern':'contrast_antonym', 'strategy':'kontras dengan lawan kata', 'template':'{token} adalah lawan dari {antonym} dalam kategori {category}'},
        {'id':'e5', 'pattern':'analogy_transport', 'strategy':'analogi via transport', 'template':'seperti {source} dalam {source_cat}, {target} dalam {target_cat}'},
    ],

    # -------------------------------------------------------------------------
    # 6. COMPREHENSION TASKS (Cara memahami)
    # -------------------------------------------------------------------------
    'comprehension_tasks': [
        {
            'id':'c1', 'context':'o1',
            'question':'apakah anak makan mangga di kebun?',
            'expected_answer':'ya', 'required_bindings':['agen','predikat','pasien','lokasi'],
            'inference_needed':False
        },
        {
            'id':'c2', 'context':'o1',
            'question':'apakah anak senang setelah makan?',
            'expected_answer':'ya', 'required_bindings':['agen','predikat'],
            'inference_needed':True, 'inference_rule':'r1'
        },
        {
            'id':'c3', 'context':'o4',
            'question':'apakah burung berada di atas?',
            'expected_answer':'ya', 'required_bindings':['agen','predikat'],
            'inference_needed':True, 'inference_rule':'r3'
        },
        {
            'id':'c4', 'context':'o1',
            'question':'apakah anak makan pisang?',
            'expected_answer':'tidak', 'required_bindings':['agen','predikat','pasien'],
            'inference_needed':False, 'contradiction_check':True
        },
        {
            'id':'c5', 'context':['o1','o7'],
            'question':'apa persamaan antara anak dan ayah?',
            'expected_answer':'keduanya adalah agen', 'required_bindings':['agen'],
            'inference_needed':True, 'integration':'compare_roles'
        },
    ],

    # -------------------------------------------------------------------------
    # 7. CONTRADICTION & ENTAILMENT (Logika)
    # -------------------------------------------------------------------------
    'logical_pairs': [
        {'id':'l1', 'stmt1':{'pasien':'batu','atribut':'besar'}, 'stmt2':{'pasien':'batu','atribut':'kecil'}, 'relation':'contradiction', 'dimension':'ukuran'},
        {'id':'l2', 'stmt1':{'pasien':'air','atribut':'panas'}, 'stmt2':{'pasien':'air','atribut':'dingin'}, 'relation':'contradiction', 'dimension':'keadaan'},
        {'id':'l3', 'stmt1':{'agen':'orang','predikat':'senang'}, 'stmt2':{'agen':'orang','predikat':'sedih'}, 'relation':'contradiction', 'dimension':'emosi'},
        {'id':'l4', 'stmt1':{'agen':'burung','predikat':'terbang'}, 'stmt2':{'agen':'burung','lokasi':'atas'}, 'relation':'entailment', 'dimension':'spatial'},
        {'id':'l5', 'stmt1':{'agen':'guru','predikat':'katakan'}, 'stmt2':{'agen':'murid','predikat':'dengar'}, 'relation':'entailment', 'dimension':'communication'},
    ],

    # -------------------------------------------------------------------------
    # 8. TEACHING EPISODES (Meta-learning: cara belajar)
    # -------------------------------------------------------------------------
    'teaching_episodes': [
        {'id':'t1', 'instruction':'pelajari bahwa makan menyebabkan senang', 'type':'induce_transform', 'data':{'from':'makan','to':'senang'}, 'verify_by':'qa7'},
        {'id':'t2', 'instruction':'pelajari bahwa terbang ke atas', 'type':'induce_transform', 'data':{'from':'terbang','to':'atas'}, 'verify_by':'qa3'},
        {'id':'t3', 'instruction':'pelajari untuk menjawab siapa dengan mencari agen', 'type':'induce_qa_strategy', 'data':{'q_type':'siapa','target_role':'agen'}, 'verify_by':'qa1'},
        {'id':'t4', 'instruction':'pelajari untuk mendeteksi kontradiksi ukuran', 'type':'induce_contradiction', 'data':{'dimension':'ukuran','antonyms':['besar','kecil']}, 'verify_by':'l1'},
    ]
}

print(f"[Dataset v2] Loaded research schema:")
print(f"  Observations: {len(fhrr_research_dataset['observations'])}")
print(f"  Reasoning patterns: {len(fhrr_research_dataset['reasoning_patterns'])}")
print(f"  QA pairs: {len(fhrr_research_dataset['qa_pairs'])}")
print(f"  Explanation templates: {len(fhrr_research_dataset['explanation_templates'])}")
print(f"  Comprehension tasks: {len(fhrr_research_dataset['comprehension_tasks'])}")
print(f"  Teaching episodes: {len(fhrr_research_dataset['teaching_episodes'])}")
