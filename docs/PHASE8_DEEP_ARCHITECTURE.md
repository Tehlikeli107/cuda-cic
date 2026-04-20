# Phase 8: Deep CIC Architecture (Calculus of Inductive Constructions)

Bu aşama, CUDA-CIC projesini basit bir hızlandırıcıdan çıkarıp, bağımlı tipler (Dependent Types), otonom bellek yönetimi ve karşılıklı tümevarımı (Mutual Induction) tamamen GPU üzerinde yerel olarak çalıştırabilen "Universal CIC Virtual Machine" seviyesine taşır.

## Sıralı İcra Planı (The Master Sequence)

### 1. Hash-Consing (Yapısal Paylaşım - Structural Sharing) ve GPU Hafıza Modeli
**Neden İlk Adım:** Bağımlı tipler (Dependent Pattern Matching) ve derin `Iota` indirgemeleri sürekli olarak yeni sentaktik ağaçlar (AST) sentezler. GPU üzerinde `malloc` yapamayacağımız ve MN (Max Nodes) sınırını anında dolduracağımız için, aynı alt-ağaçların (Örn: `x + 1`) bellek israfını önlemek şarttır.
*   **Aksiyon:** `cic_engine.cu` içindeki `ALLOC_NODE` makrosunu, lock-free bir "Thread-Local Hash Table" veya "Block-Shared Hash Table" kullanacak bir deduplication (tekilleştirme) fonksiyonuna dönüştürmek. Bir node yaratılmadan önce hash'ine bakılacak, varsa pointer'ı dönecek.

### 2. Tiplerin AST (Expression) Olarak Modellenmesi (Types as Terms)
**Neden:** Şu anki sistem tipleri `int64_t` hash'ler olarak tutuyor (`T_NAT`, `T_BOOL`). Bu "Bağımlı Tipler" (Dependent Types) için ölümcüldür. `f : (n : Nat) -> Vector n` fonksiyonunun dönüş tipi sabit bir hash değildir, `n`'e bağlı dinamik bir ağaçtır.
*   **Aksiyon:** `cic_engine.cu`'daki Type Checker modülü (`res = a2` vs.) tamamen değişecek. Bir ifadenin tipi artık 32-bit bir integer hash değil, başka bir AST düğümünün (node) indeksi olacak.

### 3. Bağımlı Desen Eşleştirme (Dependent Pattern Matching & Axiom K)
**Neden:** Tipler artık AST olduğuna göre, bir IOTA reduction gerçekleştiğinde (`App(Nat.rec motive base step, Nat.succ x)`), `motive` (Type-level fonksiyon) indirgenen değere (x) uygulanmalıdır.
*   **Aksiyon:** IOTA reduction kod bloğu, sadece minor premise'i çalıştırmakla kalmayıp, dönüş tipinin de `subst(motive, 0, x)` olduğunu Type Checker'a ispatlayacak şekilde güncellenecek. `Eq.rec` (Eşitlik recursor'ı) GPU motoruna yerel (native) donanım talimatı gibi eklenecek.

### 4. Karşılıklı İndüktif Tipler (Mutually Inductive Types) & Nested Recursion
**Neden:** `Tree` ve `List of Trees` gibi birbirini çağıran veri yapıları matematikte ve programlamada çok yaygındır.
*   **Aksiyon:** Python ortam inşa edicisi (`env_builder.py`), Lean4'ten gelen mutual blokları tespit edecek. GPU motorundaki `N_REC` yapısı, tek bir Constructor kimliğine (ID) değil, o bloğun başlangıç ofsetine göre (stride) "Mutual" atlamalar yapacak şekilde genişletilecek.

### 5. Quotient Tipleri (Bölüm Tipleri)
**Neden:** Eşdeğerlik sınıfları (Equivalence classes) üzerine kurulu matematik (`1/2 = 2/4`) yapısal eşitlik (DefEq) ile çözülemez.
*   **Aksiyon:** `Quotient.sound` GPU'ya eklenecek. İki ağacın `DefEq` kontrolü yapılırken, eğer tipleri bir Quotient tipi ise, yapısal (pointer/AST) eşitlik aramayı bırakıp eşdeğerlik ilişkisinin (equivalence relation) ispatı aranacak.

Bu adımlar sırayla CUDA kod tabanını "Derin Teknoloji" (Deep Tech) alanına iten zorlu mühendislik görevleridir. Adım 1 (Hash-Consing) ile başlıyoruz.