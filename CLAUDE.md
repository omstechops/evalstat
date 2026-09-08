# Proje Bağlamı

## Kim çalışıyor

İstatistik lisans öğrencisi + ses mühendisliği eğitimi almış. Türkiye/Bursa.
Sohbet dili Türkçe. **Kod, commit mesajları, docstring, README ve yayınlanacak
her şey İngilizce.**

## İki proje, tek omurga

Bu depo Proje A'dır. İki paralel proje var, bağımsız değiller:

- **Proje A — `evalstat`:** (bu depo) LLM/ajan değerlendirmelerinin istatistiksel geçerliliği
  için Python paketi. Güç analizi, eşleştirilmiş bootstrap, çoklu karşılaştırma
  düzeltmesi, LLM-hakem uyum ölçümü, alt grup kırılımı.
- **Proje B — `tr-speech-eval`:** (https://github.com/omstechops/tr-speech-eval)
  Türkçe konuşma sistemleri (ASR/TTS/ses ajanı)
  için değerlendirme seti + ses ajanı prototipi.

**A metodu üretir, B o metodu uygular.** B'nin her istatistiksel sonucu A ile
hesaplanır. Bu bağ projenin ana fikridir; kopmasına izin verme.

Detaylı planlar (`proje-a-eval-istatistigi.md`, `proje-b-turkce-ses-ai.md`,
`yol-haritasi.md`) yayınlanmıyor; özel çalışma deposunda kalıyor.

---

## Devretme sınırları — ÖNEMLİ

Bu projelerin değeri, sahibinin AI çıktısını **değerlendirebilmesi**. O yüzden
bazı işler bilerek devredilmez.

### Serbestçe yap
- İskele: paket yapısı, `pyproject.toml`, CI, test iskeleti, tip anotasyonu
- Sıkıcı boru hattı: toplu ses dönüşümü, SNR ekleme, yankı konvolüsyonu, batch runner
- Kütüphane keşfi ve karşılaştırması
- İngilizce redaksiyon (argüman sahibinin, cümleler senin)
- Kavram açıklama ve makale özetleme

### Yapma — önce sor
- **İstatistiksel test seçme.** Hangi testin uygun olduğuna sahibi karar verir.
  Sen uygulamasını yazarsın. Bir test önerirsen varsayımlarını (eşleştirme,
  normallik, bağımsızlık, varyans homojenliği) açıkça listele ve
  "bunlar sağlanıyor mu?" diye sor. Sessizce `t-test` çağırma.
- **Değerlendirme seti tasarımı.** Hangi akustik koşullar, hangi konuşmacı
  grupları, hangi Türkçe'ye özgü zorluklar — bu sahibinin ses mühendisliği
  bilgisi. Jenerik liste üretme.
- **Sonuç yorumu.** Raporlarda "bu veri şunu gösteriyor" cümlesini sen yazma.
  Sayıları çıkar, yorumu sahibine bırak.
- **Dinleme yargısı gerektiren hiçbir şey.** Prozodi, gecikme hissi, gürültü
  bastırmanın sesi öldürüp öldürmediği.

---

## Çalışma kuralları

- **Küçük adımlar.** Tek seferde çok dosya değiştirme. Sahibi ürettiğin her satırı
  okuyacak; okunamaz büyüklükte diff üretme.
- **Anlaşılmayan kod depoya girmez.** Alışılmadık bir şey yaparsan neden yaptığını
  yorum satırında değil, cevabında açıkla.
- **Kapsam bekçiliği.** Bir talep planların "yapmayacaklar" listesine giriyorsa
  (kendi model eğitmek, genel amaçlı eval çatısı yazmak, dashboard/SaaS, çok dilli
  platform) uygulamadan önce uyar.
- **Test önce.** İstatistiksel fonksiyonlar için bilinen sonuçlu referans vaka ile
  test yaz. Bu paketin tüm iddiası doğruluk; test edilmemiş fonksiyon yayınlanmaz.
- **Bağımlılık cimriliği.** numpy/scipy/pandas yeterli. Yeni bağımlılık eklemeden önce sor.

## Düşman gözü modu

Sahibi "bunu eleştir" veya "zayıf noktalarını bul" derse: savunma yapma, kibar olma.
Bir hakem gibi metodolojik açıkları, sızdıran varsayımları ve döngüsel akıl
yürütmeyi bul.

## Döngüsel bulaşma uyarısı

A projesinde aynı modelin hem seti tasarlaması hem hakemlik yapması hem de sonucu
yorumlaması çalışmayı geçersiz kılar. Rolleri ayrı tut ve sahibi bu ayrımı ihlal
ediyorsa hatırlat. İnsan referans etiketi adımı atlanamaz.

## Yararlı komutlar

```bash
pytest -q                 # testler
ruff check . && ruff format .
python -m build           # paketleme
```
