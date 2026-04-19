from .symupe import SyMuPe
from .symupe_local import SyMuPeLocal


class SyMuPeOnset(SyMuPeLocal):
    r"""SyMuPeOnset: a SyMuPe tokenizer with inter-onset tempos."""

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True

        self.config.additional_params["onset_tempos"] = True


class SyMuPeBeat(SyMuPe):
    r"""SyMuPeBeat: a SyMuPe tokenizer with inter-beat tempos."""

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True
        self.config.additional_params["rel_onset_dev"] = True
        self.config.additional_params["rel_perf_duration"] = True

        self.config.additional_params["bar_tempos"] = False


class SyMuPeBar(SyMuPe):
    r"""SyMuPeBar: a SyMuPe tokenizer with inter-bar tempos."""

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True
        self.config.additional_params["rel_onset_dev"] = True
        self.config.additional_params["rel_perf_duration"] = True

        self.config.additional_params["bar_tempos"] = True


class SyMuPeWindow(SyMuPeLocal):
    r"""SyMuPeBeat: a SyMuPe tokenizer with local window onset tempos."""

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True

        self.config.additional_params["use_quantized_tempos"] = self.config.additional_params.get(
            "use_quantized_tempos", False
        )
        self.config.additional_params["decode_recompute_tempos"] = False


class SyMuPeWindowRecompute(SyMuPeLocal):
    r"""SyMuPeBeat: a SyMuPe tokenizer with local window onset tempo recomputed from the tokens directly."""

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True

        self.config.additional_params["use_quantized_tempos"] = self.config.additional_params.get(
            "use_quantized_tempos", False
        )
        self.config.additional_params["decode_recompute_tempos"] = True
