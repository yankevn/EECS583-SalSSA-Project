# coding=utf8
"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxStringDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')
        ns = 'ndk' if lldbplatformutil.target_is_android() else ''
        self.namespace = 'std::__' + ns + '1'

    @add_test_categories(["libc++"])
    @expectedFailureAll(bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        is_64_bit = self.process().GetAddressByteSize() == 8

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        ns = self.namespace
        self.expect(
            "frame variable",
            substrs=[
                '(%s::wstring) wempty = L""'%ns,
                '(%s::wstring) s = L"hello world! ?????? ??????!"'%ns,
                '(%s::wstring) S = L"!!!!"'%ns,
                '(const wchar_t *) mazeltov = 0x',
                'L"?????? ??????"',
                '(%s::string) empty = ""'%ns,
                '(%s::string) q = "hello world"'%ns,
                '(%s::string) Q = "quite a long std::strin with lots of info inside it"'%ns,
                '(%s::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"'%ns,
                '(%s::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0?????? ?????????????????? ?????????????????? ??????????????? ??????"'%ns,
                '(%s::u16string) u16_string = u"????????"'%ns,
                # FIXME: This should have a 'u' prefix.
                '(%s::u16string) u16_empty = ""'%ns,
                '(%s::u32string) u32_string = U"????????????????"'%ns,
                # FIXME: This should have a 'U' prefix.
                '(%s::u32string) u32_empty = ""'%ns,
                '(%s::basic_string<unsigned char, %s::char_traits<unsigned char>, '
                '%s::allocator<unsigned char> >) uchar = "aaaaa"'%(ns,ns,ns),
        ])

        self.runCmd("n")

        TheVeryLongOne = self.frame().FindVariable("TheVeryLongOne")
        summaryOptions = lldb.SBTypeSummaryOptions()
        summaryOptions.SetCapping(lldb.eTypeSummaryUncapped)
        uncappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(uncappedSummaryStream, summaryOptions)
        uncappedSummary = uncappedSummaryStream.GetData()
        self.assertTrue(uncappedSummary.find("someText") > 0,
                        "uncappedSummary does not include the full string")
        summaryOptions.SetCapping(lldb.eTypeSummaryCapped)
        cappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(cappedSummaryStream, summaryOptions)
        cappedSummary = cappedSummaryStream.GetData()
        self.assertTrue(
            cappedSummary.find("someText") <= 0,
            "cappedSummary includes the full string")

        self.expect_expr("s", result_type=ns+"::wstring", result_summary='L"hello world! ?????? ??????!"')

        self.expect(
            "frame variable",
            substrs=[
                '(%s::wstring) S = L"!!!!!"'%ns,
                '(const wchar_t *) mazeltov = 0x',
                'L"?????? ??????"',
                '(%s::string) q = "hello world"'%ns,
                '(%s::string) Q = "quite a long std::strin with lots of info inside it"'%ns,
                '(%s::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"'%ns,
                '(%s::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0?????? ?????????????????? ?????????????????? ??????????????? ??????"'%ns,
                '(%s::u16string) u16_string = u"????????"'%ns,
                '(%s::u32string) u32_string = U"????????????????"'%ns,
                '(%s::u32string) u32_empty = ""'%ns,
                '(%s::basic_string<unsigned char, %s::char_traits<unsigned char>, '
                '%s::allocator<unsigned char> >) uchar = "aaaaa"'%(ns,ns,ns),
        ])

        if is_64_bit:
            self.expect("frame variable garbage1", substrs=['garbage1 = Summary Unavailable'])
            self.expect("frame variable garbage2", substrs=['garbage2 = Summary Unavailable'])
            self.expect("frame variable garbage3", substrs=['garbage3 = Summary Unavailable'])
            self.expect("frame variable garbage4", substrs=['garbage4 = Summary Unavailable'])
            self.expect("frame variable garbage5", substrs=['garbage5 = Summary Unavailable'])
