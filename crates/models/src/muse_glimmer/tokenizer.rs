//! Muse Glimmer's tokenizer contract: the ATEM turn markers, each one token.

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::atem::STOP_TOKENS, chat_template::atem::MARKERS],
    pinned: &[],
};
