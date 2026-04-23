import { FormEvent, useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { VideoBackground } from './components/VideoBackground';

interface SourceResult {
  rank: number;
  chunk_id: string;
  chunk_title: string;
  chunk_body: string;
  preview: string;
  source_title: string;
  source_url: string;
  chunk_index: number;
  rerank_score: number | null;
  vector_score: number | null;
  bm25_score: number | null;
  vector_rank: number | null;
  bm25_rank: number | null;
}

interface AskResponse {
  query: string;
  answer: string | null;
  retrieved_count: number;
  sources: SourceResult[];
}

interface HealthResponse {
  status: string;
  has_gemini_api_key?: boolean;
}

const DEFAULT_QUERY = 'Viêm phổi là gì?';
const GITHUB_REPO_URL = 'https://github.com/Tuancoolboy/RAG_disease_improvement';
const CHUNK_OPTIONS = [3, 5, 8, 10];
const CANDIDATE_OPTIONS = [10, 20, 30, 40];
const MISSING_GEMINI_WARNING = 'Backend chưa có GEMINI_API_KEY, nên hiện tại chỉ trả về chunks và nguồn tham khảo.';

function getApiBaseUrl(): string {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/$/, '');
  }

  if (import.meta.env.DEV) {
    if (typeof window === 'undefined') {
      return 'http://localhost:8000';
    }

    const host = window.location.hostname || 'localhost';
    return `${window.location.protocol}//${host}:8000`;
  }

  if (typeof window === 'undefined') {
    return '';
  }

  return window.location.origin;
}

function getErrorMessage(payload: unknown, fallbackStatus: number): string {
  if (typeof payload === 'string' && payload.trim()) {
    return payload;
  }

  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = payload.detail;
    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }
  }

  return `Yêu cầu thất bại (${fallbackStatus}).`;
}

export default function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [vectorTopK, setVectorTopK] = useState(20);
  const [bm25TopK, setBm25TopK] = useState(20);
  const [rerankTopK, setRerankTopK] = useState(5);
  const [includeAnswer, setIncludeAnswer] = useState(true);
  const [selectedSource, setSelectedSource] = useState('all');
  const [selectedChunk, setSelectedChunk] = useState('all');
  const [result, setResult] = useState<AskResponse | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [backendHasGeminiKey, setBackendHasGeminiKey] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth(): Promise<void> {
      try {
        const response = await fetch(`${getApiBaseUrl()}/api/health`);
        if (!response.ok) {
          return;
        }

        const payload = (await response.json().catch(() => null)) as HealthResponse | null;
        if (cancelled || !payload || payload.status !== 'ok') {
          return;
        }

        const hasGeminiKey = payload.has_gemini_api_key !== false;
        setBackendHasGeminiKey(hasGeminiKey);

        if (!hasGeminiKey) {
          setIncludeAnswer(false);
          setWarning(MISSING_GEMINI_WARNING);
        }
      } catch {
        // Ignore health probe failures and let the first submit surface any real backend error.
      }
    }

    void loadHealth();

    return () => {
      cancelled = true;
    };
  }, []);

  const sourceOptions: Array<{ key: string; label: string }> = [];
  const chunkOptions: Array<{ id: string; label: string }> = [];

  for (const source of result?.sources ?? []) {
    const sourceKey = source.source_url || source.source_title;
    if (!sourceOptions.some((item) => item.key === sourceKey)) {
      sourceOptions.push({ key: sourceKey, label: source.source_title || source.source_url || 'Nguồn không tên' });
    }

    if (!chunkOptions.some((item) => item.id === source.chunk_id)) {
      chunkOptions.push({
        id: source.chunk_id,
        label: `#${source.rank} · ${source.chunk_title || source.source_title || source.chunk_id}`,
      });
    }
  }

  const filteredSources = (result?.sources ?? []).filter((source) => {
    const matchesSource = selectedSource === 'all' || (source.source_url || source.source_title) === selectedSource;
    const matchesChunk = selectedChunk === 'all' || source.chunk_id === selectedChunk;
    return matchesSource && matchesChunk;
  });

  const selectedChunkRecord =
    selectedChunk === 'all'
      ? filteredSources[0] ?? null
      : filteredSources.find((source) => source.chunk_id === selectedChunk) ?? null;

  async function requestAsk(shouldIncludeAnswer: boolean): Promise<AskResponse> {
    const response = await fetch(`${getApiBaseUrl()}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        vector_top_k: vectorTopK,
        bm25_top_k: bm25TopK,
        rerank_top_k: rerankTopK,
        include_answer: shouldIncludeAnswer,
      }),
    });

    const payload = (await response.json().catch(() => null)) as unknown;

    if (!response.ok) {
      throw new Error(getErrorMessage(payload, response.status));
    }

    return payload as AskResponse;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setWarning(backendHasGeminiKey === false ? MISSING_GEMINI_WARNING : null);

    try {
      let response: AskResponse;
      const shouldIncludeAnswer = includeAnswer && backendHasGeminiKey !== false;

      try {
        response = await requestAsk(shouldIncludeAnswer);
      } catch (submitError) {
        const message = submitError instanceof Error ? submitError.message : 'Không gọi được backend.';
        if (shouldIncludeAnswer && message.includes('Missing Gemini API key')) {
          setBackendHasGeminiKey(false);
          setIncludeAnswer(false);
          response = await requestAsk(false);
          setWarning(MISSING_GEMINI_WARNING);
        } else {
          throw submitError;
        }
      }

      setResult(response);
      setSelectedSource('all');
      setSelectedChunk('all');
    } catch (submitError) {
      setResult(null);
      setError(submitError instanceof Error ? submitError.message : 'Không gọi được backend.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-transparent font-manrope text-brand-dark">
      <VideoBackground src="https://stream.mux.com/02gzwandixH4J534bd00JsCvlFfw6ha101WQ00C9b3sGibM.m3u8" />

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-7xl flex-col px-6 py-6 md:px-10 lg:px-12">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-full border border-black/8 bg-white/50 backdrop-blur-sm">
              <svg width="22" height="22" viewBox="0 0 23 23" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="11.5" cy="11.5" r="11.5" fill="url(#logo-gradient)" />
                <defs>
                  <radialGradient
                    id="logo-gradient"
                    cx="0"
                    cy="0"
                    r="1"
                    gradientUnits="userSpaceOnUse"
                    gradientTransform="translate(11.5 11.5) rotate(90) scale(11.5)"
                  >
                    <stop stopColor="#8FD6FF" />
                    <stop offset="0.4" stopColor="#6AB8FF" />
                    <stop offset="1" stopColor="#FFE082" />
                  </radialGradient>
                </defs>
              </svg>
            </div>
            <div>
              <p className="font-instrument-serif text-[28px] leading-none text-brand-dark">disease-rag</p>
              <p className="mt-1 text-sm text-brand-dark/70">Tìm và tóm tắt thông tin bệnh học dễ hiểu</p>
            </div>
          </div>

          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="status-pill hidden md:flex transition hover:border-black/14 hover:bg-white/82 hover:text-brand-dark"
          >
            Xem GitHub dự án
          </a>
        </header>

        <main className="flex flex-1 flex-col justify-center gap-8 py-10 md:gap-10 md:py-16">
          <motion.section
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="max-w-4xl"
          >
            <div className="status-pill mb-5 inline-flex">Website tra cứu thông tin bệnh học</div>
            <h1 className="copy-shadow max-w-4xl font-instrument-serif text-[44px] leading-[1.04] text-brand-dark md:text-[68px] lg:text-[84px]">
              Khám phá thông tin về các loại bệnh bằng tiếng Việt,
              <span className="block text-gradient-radial italic">dễ hiểu, rõ ràng và có nguồn tham khảo trực tiếp.</span>
            </h1>
            <p className="mt-5 max-w-3xl text-[17px] leading-8 text-brand-dark/78 md:text-[20px]">
              Website giúp người dùng tìm hiểu triệu chứng, nguyên nhân, cách nhận biết và các thông tin liên quan đến
              nhiều loại bệnh, đồng thời cung cấp nguồn tham khảo để đọc sâu hơn khi cần.
            </p>
          </motion.section>

          <motion.form
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.6, ease: 'easeOut' }}
            onSubmit={handleSubmit}
            className="grid gap-4"
          >
            <label className="field-shell rounded-[32px] p-5 md:p-6">
              <span className="mb-3 block text-sm font-semibold uppercase tracking-[0.24em] text-brand-dark/60">
                Câu hỏi
              </span>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="min-h-[132px] resize-y text-[18px] leading-8 text-brand-dark placeholder:text-brand-dark/40"
                placeholder="Nhập câu hỏi của bạn về bệnh học..."
              />
            </label>

            <div className="grid gap-3 xl:grid-cols-[1fr_1fr_1fr_auto]">
              <label className="field-shell field-shell--compact">
                <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-brand-dark/55">
                  Chunk dùng để trả lời
                </span>
                <select value={rerankTopK} onChange={(event) => setRerankTopK(Number(event.target.value))}>
                  {CHUNK_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option} chunk
                    </option>
                  ))}
                </select>
                <span className="mt-2 block text-xs leading-5 text-brand-dark/58">
                  Số này càng lớn thì hệ thống đọc nhiều chunk liên quan hơn trước khi đưa ra kết quả.
                </span>
              </label>

              <label className="field-shell field-shell--compact field-shell--semantic">
                <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-brand-dark/55">
                  Ứng viên vector
                </span>
                <select value={vectorTopK} onChange={(event) => setVectorTopK(Number(event.target.value))}>
                  {CANDIDATE_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option} kết quả
                    </option>
                  ))}
                </select>
                <span className="mt-2 block text-xs leading-5 text-brand-dark/58">
                  Cách này tìm các chunk có ý gần giống với câu hỏi, kể cả khi từ ngữ không hoàn toàn trùng nhau.
                </span>
              </label>

              <label className="field-shell field-shell--compact field-shell--keyword">
                <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-brand-dark/55">
                  Ứng viên BM25
                </span>
                <select value={bm25TopK} onChange={(event) => setBm25TopK(Number(event.target.value))}>
                  {CANDIDATE_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option} kết quả
                    </option>
                  ))}
                </select>
                <span className="mt-2 block text-xs leading-5 text-brand-dark/58">
                  Cách này ưu tiên các chunk có chữ hoặc cụm từ xuất hiện giống với câu hỏi.
                </span>
              </label>

              <motion.button
                whileHover={{ y: -2, opacity: 0.96 }}
                whileTap={{ scale: 0.99 }}
                type="submit"
                disabled={loading || query.trim().length < 2}
                className="flex min-h-[72px] items-center justify-center rounded-[28px] border border-black/10 bg-white/70 px-7 text-[16px] font-semibold text-brand-dark shadow-[0_18px_48px_rgba(0,0,0,0.12)] backdrop-blur-sm transition disabled:cursor-not-allowed disabled:opacity-55"
              >
                {loading ? 'Đang truy vấn...' : 'Lấy text và nguồn'}
              </motion.button>
            </div>

            <label className="inline-flex items-center gap-3 text-sm text-brand-dark/82">
              <input
                type="checkbox"
                checked={includeAnswer}
                onChange={(event) => setIncludeAnswer(event.target.checked)}
                disabled={backendHasGeminiKey === false}
                className="h-4 w-4 rounded border-black/20 bg-white/50 accent-[#212121]"
              />
              {backendHasGeminiKey === false
                ? 'Sinh câu trả lời AI đang tắt vì backend chưa có `GEMINI_API_KEY`'
                : 'Sinh câu trả lời AI nếu backend có `GEMINI_API_KEY`'}
            </label>
          </motion.form>

          {warning ? <div className="soft-panel text-sm text-amber-100">{warning}</div> : null}
          {error ? <div className="soft-panel text-sm text-rose-100">{error}</div> : null}

          <section className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
            <motion.article
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12, duration: 0.55, ease: 'easeOut' }}
              className="soft-panel"
            >
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.24em] text-brand-dark/52">Kết quả chính</p>
                  <h2 className="mt-2 text-2xl font-semibold text-brand-dark">Câu trả lời dễ đọc</h2>
                </div>
                <div className="status-pill">
                  {result ? `${result.retrieved_count} chunk đã lấy` : 'Chưa có truy vấn'}
                </div>
              </div>

              {result ? (
                result.answer ? (
                  <p className="whitespace-pre-line text-[16px] leading-8 text-brand-dark/88">{result.answer}</p>
                ) : (
                  <p className="text-[16px] leading-8 text-brand-dark/78">
                    Backend đã trả về chunks và nguồn nhưng chưa sinh câu trả lời AI. Bạn vẫn có thể đọc trực tiếp phần
                    text của từng chunk ở khung bên phải.
                  </p>
                )
              ) : (
                <p className="text-[16px] leading-8 text-brand-dark/78">
                  Gửi một câu hỏi để hiển thị phần text trả lời từ hệ RAG. Giao diện này không dùng card login hay
                  banner bán hàng nữa, chỉ tập trung vào text và dữ liệu truy xuất.
                </p>
              )}
            </motion.article>

            <motion.aside
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.16, duration: 0.55, ease: 'easeOut' }}
              className="soft-panel"
            >
              <div className="mb-4">
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-brand-dark/52">Chunk explorer</p>
                <h2 className="mt-2 text-2xl font-semibold text-brand-dark">Chọn chunk và nguồn</h2>
              </div>

              {result ? (
                <>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="field-shell field-shell--compact">
                      <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-brand-dark/55">
                        Lọc theo nguồn
                      </span>
                      <select value={selectedSource} onChange={(event) => setSelectedSource(event.target.value)}>
                        <option value="all">Tất cả nguồn</option>
                        {sourceOptions.map((option) => (
                          <option key={option.key} value={option.key}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="field-shell field-shell--compact">
                      <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-brand-dark/55">
                        Lọc theo chunk
                      </span>
                      <select value={selectedChunk} onChange={(event) => setSelectedChunk(event.target.value)}>
                        <option value="all">Tất cả chunk</option>
                        {chunkOptions.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="mt-4 rounded-[24px] border border-black/8 bg-white/62 p-5 shadow-[0_14px_36px_rgba(0,0,0,0.08)]">
                    {selectedChunkRecord ? (
                      <>
                        <div className="flex flex-wrap items-center gap-2 text-sm text-brand-dark/68">
                          <span className="status-pill">Rank #{selectedChunkRecord.rank}</span>
                          <span className="status-pill">Chunk #{selectedChunkRecord.chunk_index}</span>
                        </div>
                        <h3 className="mt-4 text-xl font-semibold text-brand-dark">
                          {selectedChunkRecord.chunk_title || 'Chunk không có tiêu đề'}
                        </h3>
                        <p className="mt-2 text-sm text-brand-dark/60">{selectedChunkRecord.source_title}</p>
                        <p className="mt-4 max-h-[280px] overflow-auto whitespace-pre-line pr-2 text-[15px] leading-7 text-brand-dark/82">
                          {selectedChunkRecord.chunk_body}
                        </p>
                      </>
                    ) : (
                      <p className="text-brand-dark/70">Chọn một chunk hoặc nguồn để đọc phần text gốc.</p>
                    )}
                  </div>
                </>
              ) : (
                <p className="text-[15px] leading-7 text-brand-dark/76">
                  Sau khi truy vấn, bạn có thể chọn riêng từng chunk hoặc chỉ giữ lại các nguồn mong muốn để đọc text.
                </p>
              )}
            </motion.aside>
          </section>

          <motion.section
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.55, ease: 'easeOut' }}
            className="soft-panel"
          >
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-brand-dark/52">Nguồn tham khảo</p>
                <h2 className="mt-2 text-2xl font-semibold text-brand-dark">Các nguồn lấy được</h2>
              </div>
              <div className="status-pill">
                {filteredSources.length > 0 ? `${filteredSources.length} mục đang hiển thị` : 'Chưa có nguồn'}
              </div>
            </div>

            {filteredSources.length > 0 ? (
              <div className="scroll-rectangle max-h-[420px] space-y-3 overflow-y-auto pr-2">
                {filteredSources.map((source) => (
                  <article
                    key={`${source.chunk_id}-${source.rank}`}
                    className="rounded-[18px] border border-black/8 bg-white/72 p-5 shadow-[0_12px_28px_rgba(0,0,0,0.07)]"
                  >
                    <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-brand-dark/48">
                      <span>Rank #{source.rank}</span>
                      <span>•</span>
                      <span>Chunk {source.chunk_index}</span>
                    </div>

                    <h3 className="mt-3 text-xl font-semibold text-brand-dark">{source.chunk_title || source.source_title}</h3>

                    <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-brand-dark/64">
                      <span>{source.source_title}</span>
                      {source.source_url ? (
                        <a
                          href={source.source_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[#245f8c] underline decoration-black/20 underline-offset-4 transition hover:text-brand-dark"
                        >
                          Mở nguồn
                        </a>
                      ) : null}
                    </div>

                    <p className="mt-4 text-[15px] leading-7 text-brand-dark/78">{source.preview}</p>
                  </article>
                ))}
              </div>
            ) : (
              <p className="text-[15px] leading-7 text-brand-dark/74">
                Chưa có nguồn hiển thị. Gửi truy vấn hoặc đổi bộ lọc `chunk` và `nguồn`.
              </p>
            )}
          </motion.section>
        </main>
      </div>
    </div>
  );
}
