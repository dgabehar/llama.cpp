#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "parsers/parsers.h"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = common_json;

namespace autoparser {

// The reasoning open tag the generation prompt actually ended with: templates that pick the tag by request (K2-Horizon:
// one tag per reasoning_effort) list the other spellings as start_alts. Falls back to the registered start tag.
static std::string opened_reasoning_start(const analyze_reasoning & r, const std::string & generation_prompt) {
    std::string best = r.start;
    size_t      pos  = r.start.empty() ? std::string::npos : generation_prompt.find(r.start);
    for (const auto & alt : r.start_alts) {
        const size_t at = generation_prompt.find(alt);
        if (at != std::string::npos && (pos == std::string::npos || at < pos)) {
            best = alt;
            pos  = at;
        }
    }
    return best;
}

static bool request_uses_tools(const generation_params & inputs) {
    return inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
}

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt            = common_chat_template_direct_apply(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = autoparser.preserved_tokens;
    data.additional_stops.insert(data.additional_stops.end(),
        autoparser.additional_stops.begin(), autoparser.additional_stops.end());
    // the tags that end the content once the reasoning is closed also stop the generation there, so the
    // dropped remainder is not decoded (and waited for) to EOS; only when the parser extracts reasoning
    if (inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE && autoparser.reasoning.mode != reasoning_mode::NONE) {
        data.stops_after_reasoning = autoparser.content.stray_ends;
        // stray_ends_no_tools are real tool-call syntax once tools are offered, so they must never
        // stop generation in that case -- only fold them in when this request has none.
        if (!request_uses_tools(inputs)) {
            data.stops_after_reasoning.insert(data.stops_after_reasoning.end(),
                autoparser.content.stray_ends_no_tools.begin(), autoparser.content.stray_ends_no_tools.end());
        }
    }

    if (inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE && !autoparser.no_empty_reply_inert.empty()) {
        data.no_empty_reply_inert = autoparser.no_empty_reply_inert;
        if (inputs.tools.is_array() && !inputs.tools.empty()) {
            data.no_empty_reply_hold = autoparser.no_empty_reply_hold;
            std::string gp = data.generation_prompt;
            gp.erase(gp.find_last_not_of(" \t\r\n") == std::string::npos ? 0 : gp.find_last_not_of(" \t\r\n") + 1);
            for (const auto & tag : autoparser.no_empty_reply_inert) {
                if (tag.compare(0, 2, "</") != 0 && gp.size() >= tag.size() &&
                    gp.compare(gp.size() - tag.size(), tag.size(), tag) == 0) {
                    data.no_empty_reply_open = true;
                }
            }
        }
    }

    std::string parser_generation_prompt = data.generation_prompt;

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !inputs.continue_msg.empty()) {
        // Build up generation prompt manually
        const auto & msg = inputs.continue_msg;

        if (!autoparser.reasoning.start.empty()) {
            const std::string opened = opened_reasoning_start(autoparser.reasoning, data.generation_prompt);
            data.generation_prompt = data.generation_prompt.substr(0, data.generation_prompt.find(opened));
            data.generation_prompt += opened + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += autoparser.reasoning.end;
            }
        }

        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = autoparser.build_parser(inputs, parser_generation_prompt);
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
            if (autoparser.tools.format.section_optional && !autoparser.tools.format.per_call_start.empty()) {
                // a bare per-call block (no section opener) is constrained the same way
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, autoparser.tools.format.per_call_start });
            }
            if (autoparser.tools.format.openai_wrapper_trigger) {
                // model emits the OpenAI function wrapper, trigger on it
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "{\"type\": \"function\"," });
            }
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs, const std::string & generation_prompt) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();

        // A grammar-less request may also get a complete Kimi-form call; the reasoning parser needs to know it
        // (a call emitted while still "thinking" ends the reasoning). Not under a required-call grammar.
        if (tools.format.kimi_fallback && tools.format.mode == tool_format::TAG_WITH_TAGGED && !has_response_format &&
            has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && jinja_caps.supports_tool_calls) {
            ctx.kimi_openers = { "<|tool_calls_section_begin|>", "<|tool_call_begin|>" };
            ctx.kimi_calls   = tools.build_kimi_calls(ctx);
        }

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                p.space() + response_format  + p.space()
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        const std::string reasoning_start = trim_whitespace(opened_reasoning_start(reasoning, generation_prompt));
        auto              prefix          = p.prefix(generation_prompt, reasoning_start);
        auto              root            = pure_content ? prefix + parser : prefix << parser;
        if (tools.format.drop_tail && ctx.extracting_reasoning) {
            // K2-Horizon: output that fits no shape above must never fail the turn (common_chat_peg_parse falls back
            // to this rule): keep the reasoning it can tell apart and hand back everything else as content.
            p.rule("lenient-fallback", prefix << (ctx.reasoning_parser + p.space() + p.content(p.rest()) + p.end()));
        }
        return root;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        const bool request_has_tools = ctx.inputs.tools.is_array() && !ctx.inputs.tools.empty();
        const bool has_implicit_ends = request_has_tools && !implicit_ends_with_tools.empty();
        if (!end.empty() && (!end_alts.empty() || has_implicit_ends)) {
            std::vector<std::string>       ends = { trim_whitespace(end) };
            std::vector<common_peg_parser> closers = { p.optspace(end) };
            for (const auto & alt : end_alts) {
                ends.push_back(trim_whitespace(alt));
                closers.push_back(p.optspace(alt));
            }
            if (has_implicit_ends) {
                // zero-width: the tag itself belongs to what follows the reasoning
                for (const auto & tag : implicit_ends_with_tools) {
                    ends.push_back(tag);
                    closers.push_back(p.peek(p.literal(tag)));
                }
            }
            auto body = p.reasoning(p.until_one_of(ends)) + p.choice(closers);
            auto opening = p.eps();
            if (start.empty()) {
                if (!ctx.kimi_calls) {
                    return p.optional(body);
                }
            } else {
                opening = p.optspace(start);
                if (!start_alts.empty()) {
                    // the request picks the open tag (K2-Horizon: by reasoning_effort), so any spelling may be the one
                    // the generation prompt opened; the generation prompt's own newline after it comes first
                    std::vector<common_peg_parser> firsts = { opening };
                    std::vector<common_peg_parser> repeats = { p.literal(trim_whitespace(start)) };
                    for (const auto & alt : start_alts) {
                        firsts.push_back(p.optspace(alt));
                        repeats.push_back(p.literal(trim_whitespace(alt)));
                    }
                    opening = p.choice(firsts) + p.zero_or_more(p.space() + p.choice(repeats));
                }
            }
            if (ctx.kimi_calls) {
                // A complete Kimi-form call inside the reasoning ends it (the call is parsed after it); a Kimi opener
                // that does not lead to a complete call is plain reasoning text, as before.
                auto stops = ends;
                stops.insert(stops.end(), ctx.kimi_openers.begin(), ctx.kimi_openers.end());
                auto kimi_closers = closers;
                kimi_closers.push_back(p.peek(*ctx.kimi_first));
                auto kimi_body = p.reasoning(p.until_one_of(stops)) + p.choice(kimi_closers);
                return p.optional(p.choice({ opening + kimi_body, opening + body }));
            }
            return p.optional(opening + body);
        }
        if (!end.empty()) {
            if (!start.empty()) {
                // Standard tag-based: optional(<think>reasoning</think>)
                return p.optional(p.optspace(start) + p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
        }
    }

    return p.eps();
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            // p.space() drops any whitespace the model leaves between the reasoning end tag
            // and the content start tag (e.g. an empty K2-Horizon reasoning block followed by
            // a formatting newline) so it never lands at the front of content.
            return ctx.reasoning_parser + p.space() + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    // stray_ends_no_tools only applies when this request has no tools -- with tools offered they're
    // real tool-call syntax, parsed by analyze_tools instead, so this path never sees them (build_parser
    // only calls analyze_content::build_parser once tools are ruled out for this request).
    std::vector<std::string> ends = stray_ends;
    if (!request_uses_tools(ctx.inputs)) {
        ends.insert(ends.end(), stray_ends_no_tools.begin(), stray_ends_no_tools.end());
    }

    if (ctx.extracting_reasoning) {
        // Same whitespace-drop as above: only when reasoning was actually parsed out, so a
        // model with no reasoning tags at all keeps any leading whitespace it genuinely emits.
        if (!ends.empty()) {
            return ctx.reasoning_parser + p.space() + p.content(p.until_one_of(ends)) + p.optional(p.rest()) + p.end();
        }
        return ctx.reasoning_parser + p.space() + p.content(p.rest()) + p.end();
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.eps();
    if (format.section_start.empty() && !format.per_call_start.empty()) {
        auto single_tool_parser = p.standard_json_tools(
            format.per_call_start, format.per_call_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
        tools_parser = p.trigger_rule("tool-calls", p.one_or_more(single_tool_parser + p.space()));
    } else {
        tools_parser = p.standard_json_tools(
            format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
    }

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    return ctx.reasoning_parser + p.optional(p.content(p.until(tool_start))) + tools_parser + p.end();
}

common_peg_parser analyze_tools::build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                                    const common_peg_parser & call_id_section, bool have_call_id,
                                                    const common_peg_parser & args,
                                                    std::optional<common_peg_parser> atomic_peek) const {
    auto              open           = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix);
    bool              matched_atomic = false;
    common_peg_parser func_parser    = p.eps();

    if (!function.args_separator.empty()) {
        open = open + p.space() + p.literal(function.args_separator);
    }

    if (!function.name_suffix.empty()) {
        func_parser    = open + call_id_section + p.space() + args;
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser    = p.atomic(open + call_id_section) + p.space() + args;
        matched_atomic = true;
    } else if (atomic_peek.has_value()) {
        func_parser    = p.atomic(open + call_id_section + p.space() + *atomic_peek) + args;
        matched_atomic = true;
    } else {
        func_parser = open + call_id_section + p.space() + args;
    }

    if (!function.close.empty()) {
        func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
    } else if (!format.per_call_end.empty()) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        // Laguna (v4): the model may emit whitespace between the last </arg_value> and
        // </tool_call> even though the template renders them tight. Tolerate optional
        // leading space in the close lookahead so the tool call still closes.
        auto close_peek = arguments.tolerate_intertag_whitespace
                              ? p.peek(p.space() + p.literal(format.per_call_end))
                              : p.peek(p.literal(format.per_call_end));
        if (format.section_optional) {
            close_peek = p.peek(p.space() + p.choice({ p.literal(format.per_call_end), p.literal(format.section_end),
                                                        p.literal(format.per_call_start) }));
        }
        func_parser = func_parser + p.tool_close(close_peek);
    } else {
        func_parser = func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = p.atomic(func_parser);
    }
    return func_parser;
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto   schema = common_chat_tool_parameters(func);

        // Build call_id parser based on position (if supported)
        bool have_call_id = false;
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
            have_call_id = true;
        }
        auto args_parser = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));
        if (!arguments.start.empty()) {
            args_parser = p.literal(arguments.start) + args_parser;
        }
        if (!arguments.end.empty()) {
            args_parser = args_parser + p.literal(arguments.end);
        }

        auto atomic_peek = !arguments.start.empty() ? std::optional(p.peek(p.literal(arguments.start))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    const bool   parallel    = inputs.parallel_tool_calls || format.always_parallel;

    auto until_suffix = p.rule("until-suffix", p.until(arguments.value_suffix));

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func = tool.at("function");
        std::string  name = func.at("name");

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        foreach_parameter(func, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param.name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (param.schema->may_be_string() ?
                                p.ac(p.tool_arg_string_value(until_suffix) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)), arguments.value_suffix) :
                                (p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + name + "-arg-" + param.name + "-schema", doc, *param.schema)) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)))));

            auto named_arg = p.rule("tool-" + name + "-arg-" + param.name, arg);
            if (param.required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        });

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        if (!arguments.start.empty()) {
            args_seq = p.literal(arguments.start) + args_seq;
        }
        if (!arguments.end.empty()) {
            args_seq = args_seq + p.literal(arguments.end);
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            have_call_id = true;
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        auto atomic_peek = (!arguments.name_prefix.empty() && !required_parsers.empty()) ?
            std::optional(p.peek(p.literal(arguments.name_prefix))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        // a block whose own end tag is missing still ends at the section end or at the next call
        auto call_end = format.section_optional ?
            p.choice({ p.literal(format.per_call_end), p.peek(p.literal(format.section_end)), p.peek(p.literal(format.per_call_start)) }) :
            p.literal(format.per_call_end);
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + call_end;
        if (parallel) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call) + p.space());
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.space());
        }
        if (!format.section_start.empty()) {
            if (format.section_optional) {
                // a bare per-call block is a call too, and so is a section that is never closed
                tool_calls = p.trigger_rule("tool-calls",
                                            p.optional(p.literal(format.section_start) + p.space()) + tool_calls + p.space() +
                                                p.optional(p.literal(format.section_end) + p.space()));
            } else {
                tool_calls = p.trigger_rule("tool-calls",
                                            p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                                (format.section_end.empty() ? p.end() : p.literal(format.section_end) + p.space()));
            }
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

    const auto tool_calls_native = tool_calls;
    if (ctx.kimi_calls) {
        // outside the trigger rule on purpose: the lazy grammar only ever constrains the native form
        tool_calls = p.choice({ tool_calls, *ctx.kimi_calls });
    }

    const auto tool_calls_core = tool_calls;
    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    if (format.drop_tail) {
        // the content also ends at a bare call block, a Kimi-form call and the stray tags the model tacks on
        // (a second reasoning close tag, ...)
        std::vector<std::string> openers = { trigger_marker };
        if (!format.per_call_start.empty()) {
            openers.push_back(format.per_call_start);
        }
        openers.insert(openers.end(), ctx.kimi_openers.begin(), ctx.kimi_openers.end());
        std::vector<std::string> stops = openers;
        if (ctx.content) {
            stops.insert(stops.end(), ctx.content->stray_ends.begin(), ctx.content->stray_ends.end());
        }
        // Whatever follows a call is dropped. Where no call is attempted, a stray tag and everything after it is
        // dropped; a call that fails to parse is not dropped -- the parse fails and lenient-fallback keeps the text.
        // stray tags before any content (a repeated close tag right after the reasoning) are skipped, not an end
        std::vector<common_peg_parser> strays;
        if (ctx.content) {
            for (const auto & e : ctx.content->stray_ends) {
                strays.push_back(p.literal(e));
            }
        }
        auto lead = strays.empty() ? p.eps() : p.zero_or_more(p.space() + p.choice(strays));
        std::vector<common_peg_parser> attempts;
        for (const auto & o : openers) {
            attempts.push_back(p.literal(o));
        }
        auto tail        = p.optional(p.rest());
        auto tools_part  = tool_calls_core + tail;
        if (!require_tools) {
            tools_part = p.choice({ tools_part, p.negate(p.choice(attempts)) + tail });
        }
        if (!ctx.kimi_openers.empty()) {
            // a Kimi opener that never became a call, followed by the native call
            std::vector<std::string> native_openers = { trigger_marker };
            if (!format.per_call_start.empty()) {
                native_openers.push_back(format.per_call_start);
            }
            tools_part = p.choice({ tools_part, p.until_one_of(native_openers) + tool_calls_native + tail });
        }
        // a stray tag ends the content, but a call after it (a garbled restart that still calls) is a call
        auto skip = p.optional(p.until_one_of(openers));
        return ctx.reasoning_parser + lead + p.space() + p.optional(p.content(p.until_one_of(stops))) + skip + tools_part + p.end();
    }
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

common_peg_parser analyze_tools::build_kimi_calls(parser_build_context & ctx) const {
    auto &       p      = ctx.p;
    const auto & inputs = ctx.inputs;

    // `<|tool_call_begin|>[functions.]name[:N][<|tool_call_argument_begin|>|<|sep|>]{json}<|tool_call_end|>`
    common_peg_parser tool_choice = p.choice();
    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto   schema = common_chat_tool_parameters(func);

        auto open = p.tool_open(p.optional(p.literal("functions.")) + p.tool_name(p.literal(name)) +
                                p.optional(p.literal(":") + p.chars("0-9", 1, -1)));
        auto sep  = p.choice({ p.literal("<|tool_call_argument_begin|>"), p.literal("<|sep|>"), p.eps() });
        auto args = p.tool_args(p.schema(p.json(), "kimi-tool-" + name + "-schema", schema));
        tool_choice |= p.rule("kimi-tool-" + name,
                              p.atomic(open + p.space() + sep + p.space() + args) + p.space() +
                                  p.tool_close(p.literal("<|tool_call_end|>")));
    });

    auto call  = p.literal("<|tool_call_begin|>") + p.space() + tool_choice + p.space();
    auto calls = p.one_or_more(call);
    // only the first call is looked ahead at by the reasoning parser: waiting for the whole rest would stall it
    ctx.kimi_first = p.choice({ p.literal("<|tool_calls_section_begin|>") + p.space() + call, call });
    return p.choice({ p.literal("<|tool_calls_section_begin|>") + p.space() + calls +
                          p.optional(p.literal("<|tool_calls_section_end|>") + p.space()),
                      calls });
}

}  // namespace autoparser
