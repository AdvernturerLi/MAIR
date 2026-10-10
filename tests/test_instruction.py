from instruction import InstructionSpec, parse_keep, parse_only


def test_keep_haze_but_remove_rain():
    inst = InstructionSpec.parse("保留雾但去雨")
    assert inst.keep == {"haze"}
    assert inst.should_process("haze") is False
    assert inst.should_process("rain") is True


def test_except_for_haze():
    inst = InstructionSpec.parse("Remove all degradations except for the haze")
    assert inst.keep == {"haze"}
    assert inst.should_process("haze") is False
    assert inst.should_process("noise") is True


def test_only_rain():
    inst = InstructionSpec.parse("Help me only remove the rain in the picture")
    assert inst.only == {"rain"}
    assert inst.should_process("rain") is True
    assert inst.should_process("haze") is False
    assert inst.should_process("noise") is False


def test_default_restore_all():
    inst = InstructionSpec.parse("Restore the image")
    assert inst.keep == set()
    assert inst.only is None
    assert inst.should_process("rain") is True
    assert inst.should_process("noise") is True


def test_empty_instruction():
    inst = InstructionSpec.parse("")
    assert inst.keep == set()
    assert inst.only is None
    assert inst.should_process("rain") is True


def test_keep_haze_and_rain():
    inst = InstructionSpec.parse("保留雾和雨")
    assert inst.keep == {"haze", "rain"}


def test_keep_noise_but_remove_haze():
    inst = InstructionSpec.parse("去雾但保留噪声")
    assert inst.keep == {"noise"}
    assert inst.should_process("noise") is False
    assert inst.should_process("haze") is True


def test_parse_only_none_when_no_marker():
    assert parse_only("保留雾但去雨") is None


def test_only_keep_haze():
    # "只保留雾" = 只保留雾、处理其余，绝不能反向成"只处理雾"
    inst = InstructionSpec.parse("只保留雾")
    assert inst.keep == {"haze"}
    assert inst.only is None
    assert inst.should_process("haze") is False
    assert inst.should_process("rain") is True


def test_only_process_rain():
    inst = InstructionSpec.parse("只去雨")
    assert inst.only == {"rain"}
    assert inst.should_process("rain") is True
    assert inst.should_process("haze") is False
