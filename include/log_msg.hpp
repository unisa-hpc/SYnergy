#pragma once
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>


namespace synergy::log {

    enum class LogLevel { None, Info, Debug };

    inline LogLevel get_log_level_from_env() {
        const char* env = std::getenv("SYNERGY_LOG");
        if (!env) return LogLevel::None;

        std::string val(env);
        if (val == "info") return LogLevel::Info;
        if (val == "debug") return LogLevel::Debug;

        return LogLevel::None;
    }

    inline void synergy_log(LogLevel level, const std::string& msg) {
        static LogLevel current_level = get_log_level_from_env();

        if (current_level == LogLevel::None) return;
        if (level == LogLevel::Info && (current_level == LogLevel::Info || current_level == LogLevel::Debug))
            std::cout << "[INFO] " << msg << std::endl;
        else if (level == LogLevel::Debug && current_level == LogLevel::Debug)
            std::cout << "[DEBUG] " << msg << std::endl;
    }
}