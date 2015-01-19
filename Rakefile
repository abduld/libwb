require 'English'

task :default do
    system "make"
    fail "Make failed." unless $CHILD_STATUS.exitstatus == 0
end
