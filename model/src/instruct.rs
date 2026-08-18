//! The chat aspect's registry: `arch_name` in, an [`Instruct`] out.
//!
//! The *vocabulary* — the `Instruct` trait, the decoders and their events —
//! lives in `pie-model-common`, because every generation implements it and a
//! generation crate cannot depend on the registry that dispatches to it. What
//! is left here is the dispatch itself.

pub use pie_model_common::instruct::*;

use pie_model_qwen_3::chat::{CoderSchema, ToolDialect};
use pie_tokenizer::Tokenizer;
use std::sync::Arc;

/// Whether this deployment is a Qwen3-**Coder** checkpoint, which shares the
/// thinking models' architecture and vocabulary but has **no thinking channel**.
///
/// The name is the only signal there is, and that is a finding rather than a
/// shortcut. `Qwen3-Coder-30B-A3B-Instruct` and the *thinking* `Qwen3-30B-A3B`
/// are both `Qwen3MoeForCausalLM`, so the arch stem cannot separate them; both
/// carry `<think>` (151667) in vocab, so the tokenizer cannot either; and the
/// chat template that could is not in the artifact — this checkpoint keeps it
/// in a separate `chat_template.jinja` the build never imported, so
/// `chat_template` appears zero times in the `.zt`.
///
/// What it costs to get wrong is not a formatting nit. `cue_no_think()` ends the
/// generation prompt with `<think>\n\n</think>\n\n`, and a model that never saw
/// those tokens in training answers **nothing at all**. Measured on mlx-lm, same
/// checkpoint, same prompt, only the cue changed:
///
/// ```text
/// <|im_start|>assistant\n                        -> "1, yte, 3, 4, 5, 6, 7, 8, "
/// <|im_start|>assistant\n<think>\n\n</think>\n\n -> ""     (immediate EOS)
/// ```
///
/// Lineage is checked first and wins: a Qwen3.5/3.6 checkpoint is a thinking
/// model whatever its deployment name says, and separators are stripped so
/// `qwen3_5`, `qwen3_5moe`, `qwen3.6` and `qwen3-next` all match. Both haystacks
/// include the architecture because a deployment may be named anything, and the
/// substring tests are chosen to hold under either spelling — the arch stem
/// drops the underscores an HF `model_type` would have, and keying on that
/// spelling is exactly how this registry once dropped every tool schema for Qwen
/// MoE models.
/// `pub` so the renderer-parity harness
/// (`integrations/opencode/parity/render-tokens`) decides the dialect with
/// THIS predicate rather than its own copy. A parity check that guesses the
/// dialect independently validates a prompt the server never renders — which
/// is the exact failure it exists to catch.
pub fn is_coder_lineage(arch_name: &str, model_name: &str) -> bool {
    let hay = format!("{model_name} {arch_name}").to_lowercase();
    let squashed: String = hay.chars().filter(|c| !matches!(c, '_' | '.' | '-')).collect();
    if squashed.contains("qwen35") || squashed.contains("qwen36") || squashed.contains("qwen3next")
    {
        return false;
    }
    squashed.contains("coder")
}

/// Which tool dialect this checkpoint speaks.
///
/// Three answers, not two, and the third is why this replaced a boolean.
/// Qwen3.5/3.6 is a HYBRID: JSON schemas like Qwen3, nested-XML calls like
/// Qwen3-Coder. Read from the checkpoints' own `chat_template.jinja`:
///
/// ```text
///   Qwen3-8B-4bit              <tools> holds JSON, calls are JSON  -> Hermes
///   Qwen3-Coder-30B-A3B        <tools> holds XML,  calls are XML   -> Coder
///   Qwen3.6-35B-A3B            <tools> holds JSON, calls are XML   -> Qwen35Xml
/// ```
///
/// ## Two signals, because one is not enough
///
/// The 3.5/3.6 row keys on the ARCHITECTURE, which is the durable signal and
/// the one upstream uses: `qwen3_5_moe` is a fact about the checkpoint, while
/// a deployment name is whatever an operator typed.
///
/// The Coder row cannot. `Qwen3-Coder-30B-A3B-Instruct` and the *thinking*
/// `Qwen3-30B-A3B` are both `Qwen3MoeForCausalLM`, both carry `<think>` in
/// vocab, and this checkpoint ships no `_name_or_path` — so the arch stem
/// cannot separate them and the name is the only signal there is. Upstream's
/// registry drops the name entirely and would hand Qwen3-Coder the thinking
/// cue, which that checkpoint answers with an immediate EOS. Hence: arch first
/// for the row that has one, name second for the row that does not.
pub fn tool_dialect(arch_name: &str, model_name: &str) -> ToolDialect {
    let squash = |s: &str| -> String {
        s.to_lowercase().chars().filter(|c| !matches!(c, '_' | '.' | '-')).collect()
    };
    // The arch stem alone: a DEPLOYMENT called "qwen3.6-bench" served from a
    // Qwen3 checkpoint is a Qwen3, and asking the name here would say otherwise.
    let arch = squash(arch_name);
    if arch.contains("qwen35") || arch.contains("qwen36") {
        return ToolDialect::Qwen35Xml;
    }
    if is_coder_lineage(arch_name, model_name) {
        return ToolDialect::Coder;
    }
    ToolDialect::Hermes
}

/// Create the appropriate instruct implementation for the given architecture.
///
/// This match is the chat aspect's registry: `model_type` in, implementation
/// out, with every N:1 reuse (nemotron_h speaking ChatML, deepseek_v4 speaking
/// R1) stated as its own arm rather than implied by a directory. The rows
/// dispatch on the *model type*; the generation crates only hold the
/// implementations.
///
/// ## Two spellings reach this function, and only one of them is a model type
///
/// The string the engine passes is the driver's arch *stem*: `architectures[0]`
/// lowercased with its task suffix removed (`arch_stem` in `driver/metal`, and
/// its Rust twin in `worker/src/embedded_driver.rs`). CamelCase word boundaries
/// carry no separator through that, so `Qwen3_5MoeForConditionalGeneration`
/// arrives as `qwen3_5moe` — NOT as the HF model type `qwen3_5_moe` this
/// registry was written against. The mismatch falls to the `_` arm, whose
/// `has_tools: false` makes `equip_after_system` drop every tool schema and
/// `has_thinking: false` stops the think channel being handled: chat still
/// renders, the model still answers, and tool calling is simply gone with no
/// error anywhere. Observed serving Qwen3.6-35B-A3B, where a tools request and
/// a tools-free one rendered to the identical prompt-token count.
///
/// So the qwen rows carry BOTH spellings. Stems are listed beside the model
/// types they correspond to rather than normalized on the fly — a spelling that
/// reaches a registry is data, and guessing at underscore placement is how the
/// silent version of this bug comes back.
pub fn create(arch_name: &str, model_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use pie_model_qwen_3::chat::{ChatMLConfig, QwenInstruct};

    match arch_name {
        // model types …
        "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_moe"
        | "qwen3_vl" | "qwen3_vl_text"
        // … and the driver stems for the same releases: Qwen3MoeForCausalLM,
        // Qwen3_5Moe{ForCausalLM,ForConditionalGeneration},
        // Qwen3VLForConditionalGeneration.
        | "qwen3moe" | "qwen3_5moe" | "qwen3vl" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                // Every row here is a thinking model EXCEPT Qwen3-Coder, which
                // shares this arch stem and answers nothing when cued with an
                // empty think block. See `is_coder_lineage`.
                has_thinking: !is_coder_lineage(arch_name, model_name),
                has_tools: true,
                // Thinking and dialect were one signal until Qwen3.6, which
                // thinks AND speaks XML -- the pairing a single predicate could
                // not express. They are two questions now; see `tool_dialect`.
                tool_dialect: tool_dialect(arch_name, model_name),
                // Qwen3 and Qwen3-Coder lead with the caller's system message;
                // Qwen3.5/3.6 leads with the tools block. One arm serves all
                // three, so this follows the dialect rather than repeating the
                // lineage test.
                system_before_tools: !matches!(
                    tool_dialect(arch_name, model_name),
                    ToolDialect::Qwen35Xml
                ),
                // Qwen3.5/3.6 writes a reasoning block on every post-query
                // assistant turn; Qwen3 writes one only when the turn actually
                // carried reasoning.
                empty_reasoning_header: matches!(
                    tool_dialect(arch_name, model_name),
                    ToolDialect::Qwen35Xml
                ),
                // Qwen3.5/3.6 opens the model's turn INSIDE a reasoning block;
                // Qwen3 opens it bare and only the thinking-OFF branch adds
                // anything. An empty suffix here made every thinking-mode cue
                // start from a position the checkpoint never trains at -- and
                // it went unnoticed because no guest could reach the
                // thinking-on cue until `cue(thinking)` existed.
                generation_suffix: if matches!(
                    tool_dialect(arch_name, model_name),
                    ToolDialect::Qwen35Xml
                ) {
                    "<think>\n"
                } else {
                    ""
                },
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                // Coder moves the newline to the back of the block.
                tool_response_trailing_newline: matches!(
                    tool_dialect(arch_name, model_name),
                    ToolDialect::Coder
                ),
                // The variant every redistributed Qwen3-Coder checkpoint
                // carries -- mlx-community's and unsloth's alike. Qwen's own
                // repo publishes a revised file; `CoderSchema::QwenMain`
                // renders that one.
                coder_schema: CoderSchema::Shipped,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "nemotron_h" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: false,
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                generation_suffix: "<think>\n",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
                coder_schema: CoderSchema::Shipped,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "qwen2" => Arc::new(pie_model_qwen_2::chat::new(tokenizer)),
        "llama2" => Arc::new(pie_model_llama_2::chat::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(pie_model_llama_3::chat::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" | "deepseek_v4" => {
            Arc::new(pie_model_deepseek_r1::chat::R1Instruct::new(tokenizer))
        }
        "kimi_k2" | "kimi_k25" | "kimi_k3" => {
            Arc::new(pie_model_kimi_k2::chat::KimiInstruct::new(tokenizer))
        }
        "glm_moe_dsa" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
                coder_schema: CoderSchema::Shipped,
                stop_tokens: &["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"],
            },
        )),
        "gptoss" | "gpt_oss" => Arc::new(pie_model_gpt_oss::chat::GptOssInstruct::new(tokenizer)),
        "gemma2" => Arc::new(pie_model_gemma_2::chat::GemmaInstruct::new(tokenizer)),
        "gemma3" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3,
        )),
        "gemma3_text" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3Text,
        )),
        "gemma3n" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3n,
        )),
        "gemma3n_text" => Arc::new(pie_model_gemma_3::chat::Gemma3Instruct::for_variant(
            tokenizer,
            pie_model_gemma_3::chat::Gemma3Variant::Gemma3nText,
        )),
        "gemma4" => Arc::new(pie_model_gemma_4::chat::Gemma4Instruct::for_variant(
            tokenizer,
            pie_model_gemma_4::chat::Gemma4Variant::Gemma4,
        )),
        "gemma4_text" => Arc::new(pie_model_gemma_4::chat::Gemma4Instruct::for_variant(
            tokenizer,
            pie_model_gemma_4::chat::Gemma4Variant::Gemma4Text,
        )),
        "mistral3" | "ministral3" => {
            Arc::new(pie_model_mistral_3::chat::MistralInstruct::new(tokenizer))
        }
        "olmo2" => Arc::new(pie_model_olmo_2::chat::Olmo2Instruct::new(tokenizer)),
        "olmo3" => Arc::new(pie_model_olmo_3::chat::OlmoInstruct::new(tokenizer)),
        "phi3" => Arc::new(pie_model_phi_3::chat::Phi3Instruct::new(tokenizer)),
        _ => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: false,
                has_tools: false,
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
                coder_schema: CoderSchema::Shipped,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary with the ChatML control tokens the qwen instruct emits, so
    /// a rendered prompt is comparable across arch strings.
    fn tok() -> Arc<Tokenizer> {
        let v: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "system",
            "\n",
            "user",
            "assistant",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tools>",
            "</tools>",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    /// The task suffix the driver's `arch_stem` strips off `architectures[0]`.
    /// Mirrors `worker/src/embedded_driver.rs` so the two cannot drift apart
    /// silently — which is exactly how the bug below shipped.
    fn arch_stem(architectures_0: &str) -> String {
        let low = architectures_0.to_lowercase();
        low.strip_suffix("forconditionalgeneration")
            .or_else(|| low.strip_suffix("forcausallm"))
            .unwrap_or(&low)
            .to_string()
    }

    /// Does this deployment get the plain generation cue, or one with an empty
    /// think block appended?
    ///
    /// Asked through the public surface rather than by reading `has_thinking`,
    /// for the reason `renders_tools` gives: the cue is the thing that breaks,
    /// so the cue is the thing to assert on. A Coder deployment must get the
    /// SAME bytes from `cue_no_think()` as from `cue()` — anything else and the
    /// model answers with nothing.
    fn cue_is_plain(arch: &str, name: &str) -> bool {
        let i = create(arch, name, tok());
        i.cue_no_think() == i.cue()
    }

    #[test]
    fn qwen3_coder_gets_the_plain_cue_under_either_arch_spelling() {
        for arch in ["qwen3_moe", "qwen3moe"] {
            for name in [
                "mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit",
                "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                "qwen3-coder",
                "CODER-30B",
            ] {
                assert!(
                    cue_is_plain(arch, name),
                    "{name} on {arch} was cued with a think block it cannot answer"
                );
            }
        }
    }

    #[test]
    fn thinking_qwen3_still_closes_the_think_channel() {
        // The regression this must not cause: Qwen3-30B-A3B shares Coder's
        // arch stem, and it DOES think. Only the name separates them.
        for (arch, name) in [
            ("qwen3moe", "mlx-community--Qwen3-30B-A3B-4bit"),
            ("qwen3", "Qwen--Qwen3-0.6B-optimized"),
            ("qwen3", ""),
        ] {
            assert!(!cue_is_plain(arch, name), "{name} on {arch} lost its think cue");
        }
    }

    #[test]
    fn qwen3_5_lineage_outranks_a_coder_deployment_name() {
        // A 3.5/3.6 checkpoint is a thinking model whatever it was deployed as,
        // and the separators in the release spelling must not decide it.
        for name in [
            "mlx-community--Qwen3.6-35B-A3B-4bit",
            "qwen3_5_moe-coder-bench",
            "Qwen3-Next-80B-coder",
        ] {
            assert!(
                !cue_is_plain("qwen3_5moe", name),
                "{name} was demoted to the non-thinking cue by its deployment name"
            );
        }
    }

    /// The pairing the old single predicate could not express.
    ///
    /// `tool_dialect` and `has_thinking` were both `is_coder_lineage`, on the
    /// rule that a checkpoint with no thinking channel is the Coder release and
    /// the Coder release speaks XML. Qwen3.6 is a THINKING model that speaks
    /// XML, so under the coupling it was served Hermes JSON while its own
    /// `chat_template.jinja` mandates `<tool_call><function=NAME><parameter=P>`
    /// unconditionally.
    ///
    /// And the dialect is three-valued, not two: Qwen3.6 writes JSON schemas
    /// with XML calls, so serving it as `Coder` renders `<function><name>`
    /// schema blocks its template never writes. Each row's dialect is read out
    /// of that checkpoint's own template, so the table cannot drift from what
    /// is shipped.
    #[test]
    fn dialect_and_thinking_are_independent_for_qwen3_5_lineage() {
        // (arch stem, deployment name, dialect, thinks)
        for (arch, name, dialect, thinks) in [
            // Plain Qwen3: JSON schemas, JSON calls, and it thinks.
            ("qwen3", "mlx-community--Qwen3-8B-4bit", ToolDialect::Hermes, true),
            // Coder: XML schemas, XML calls, and it does NOT think. The old
            // coupling's whole case.
            (
                "qwen3moe",
                "mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit",
                ToolDialect::Coder,
                false,
            ),
            // Qwen3.6: JSON schemas, XML calls, AND it thinks -- neither of the
            // rows above, which is why the dialect stopped being a boolean.
            (
                "qwen3_5moe",
                "mlx-community--Qwen3.6-35B-A3B-4bit",
                ToolDialect::Qwen35Xml,
                true,
            ),
            ("qwen3_5moe", "Qwen--Qwen3.6-35B-A3B", ToolDialect::Qwen35Xml, true),
        ] {
            assert_eq!(
                tool_dialect(arch, name),
                dialect,
                "{name} on {arch}: wrong tool dialect -- the prompt would be \
                 rendered in one dialect and parsed in the other"
            );
            assert_eq!(
                !cue_is_plain(arch, name),
                thinks,
                "{name} on {arch}: wrong thinking channel"
            );
        }
    }

    /// The row upstream's registry does not have.
    ///
    /// `dev-sslee` keys the dialect on `arch_name` alone and drops `model_name`
    /// from `create()` entirely, which is cleaner for 3.5/3.6 and loses
    /// Qwen3-Coder: it is `qwen3_moe`, so an arch-only registry hands it
    /// Qwen3's thinking cue and the JSON dialect. The name is the only signal
    /// that separates it, so the name is still asked -- but only after the arch
    /// has had its say.
    #[test]
    fn arch_decides_before_the_name_does() {
        // A Qwen3 checkpoint DEPLOYED under a 3.6-ish name is still a Qwen3.
        assert_eq!(tool_dialect("qwen3", "qwen3.6-bench"), ToolDialect::Hermes);
        // A 3.5-architected checkpoint deployed under a coder-ish name is still
        // Qwen3.5 -- arch wins, and it is not Coder.
        assert_eq!(tool_dialect("qwen3_5moe", "qwen3_5_moe-coder-bench"), ToolDialect::Qwen35Xml);
        // And Coder still resolves, which an arch-only registry cannot do.
        assert_eq!(
            tool_dialect("qwen3moe", "Qwen3-Coder-30B-A3B-Instruct"),
            ToolDialect::Coder
        );
    }

    /// Does this arch string reach an instruct that renders tool schemas?
    ///
    /// Asked through the public surface rather than by reading `has_tools`:
    /// dropping the schemas is precisely the failure, so the test asserts on
    /// the rendering, not on the flag that governs it.
    fn renders_tools(arch: &str) -> bool {
        let schema = r#"{"name":"get_time","description":"t","parameters":{}}"#.to_string();
        let with = create(arch, "", tok()).equip_after_system(None, &[schema]);
        let without = create(arch, "", tok()).equip_after_system(None, &[]);
        with != without
    }

    /// Every qwen release the driver can hand us must reach the tool-capable
    /// instruct — through EITHER spelling.
    ///
    /// The engine passes the driver's arch stem, not the HF model type, and
    /// CamelCase word boundaries vanish in it: `Qwen3_5MoeForConditionalGeneration`
    /// arrives as `qwen3_5moe`, which used to miss `qwen3_5_moe` and fall to
    /// the tools-less default arm. Nothing failed — the model just never saw a
    /// tool schema again. Serving Qwen3.6-35B-A3B, a request with tools and one
    /// without rendered to the identical prompt-token count.
    #[test]
    fn every_qwen_release_reaches_the_tool_capable_instruct() {
        for architectures_0 in [
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",                // Qwen3-Coder-30B-A3B
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration", // Qwen3.6-35B-A3B
            "Qwen3VLForConditionalGeneration",
        ] {
            let stem = arch_stem(architectures_0);
            assert!(
                renders_tools(&stem),
                "{architectures_0} -> arch stem {stem:?} does not reach a tool-capable \
                 instruct: tool schemas would be dropped silently"
            );
        }
        // The model types the same releases report, which is what this
        // registry was originally written against. Both spellings, forever.
        for model_type in [
            "qwen3",
            "qwen3_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_vl",
        ] {
            assert!(renders_tools(model_type), "model type {model_type:?} lost its tools");
        }
    }

    /// The default arm stays tools-less: an unknown architecture gets a prompt
    /// that renders, not a guess at a tool dialect it may not speak.
    #[test]
    fn an_unknown_architecture_still_falls_through_to_plain_chatml() {
        assert!(!renders_tools("something_nobody_has_registered"));
    }
}
