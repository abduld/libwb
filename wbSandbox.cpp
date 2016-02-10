

#include <wb.h>

#ifdef WB_USE_SANDBOX

#define _GNU_SOURCE 1
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include <sys/prctl.h>
#ifndef PR_SET_NO_NEW_PRIVS
#define PR_SET_NO_NEW_PRIVS 38
#endif

#include <linux/unistd.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>

#ifndef SECCOMP_MODE_FILTER
#define SECCOMP_MODE_FILTER 2         /* uses user-supplied filter. */
#define SECCOMP_RET_KILL 0x00000000U  /* kill the task immediately */
#define SECCOMP_RET_TRAP 0x00030000U  /* disallow and force a SIGSYS */
#define SECCOMP_RET_ALLOW 0x7fff0000U /* allow */
struct seccomp_data {
  int nr;
  __u32 arch;
  __u64 instruction_pointer;
  __u64 args[6];
};
#endif
#ifndef SYS_SECCOMP
#define SYS_SECCOMP 1
#endif

#define syscall_nr (offsetof(struct seccomp_data, nr))
#define arch_nr (offsetof(struct seccomp_data, arch))

#if defined(__i386__)
#define REG_SYSCALL REG_EAX
#define ARCH_NR AUDIT_ARCH_I386
#elif defined(__x86_64__)
#define REG_SYSCALL REG_RAX
#define ARCH_NR AUDIT_ARCH_X86_64
#else
#warning "Platform does not support seccomp filter yet"
#define REG_SYSCALL 0
#define ARCH_NR 0
#endif

#define VALIDATE_ARCHITECTURE                                                  \
  BPF_STMT(BPF_LD + BPF_W + BPF_ABS, arch_nr)                                  \
  , BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, ARCH_NR, 1, 0),                        \
      BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_KILL)

#define EXAMINE_SYSCALL BPF_STMT(BPF_LD + BPF_W + BPF_ABS, syscall_nr)

#define ALLOW_SYSCALL(name)                                                    \
  BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, __NR_##name, 0, 1)                       \
  , BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_ALLOW)

#define KILL_PROCESS BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_KILL)

#ifdef WB_USE_SANDBOX_DEBUG

static const char *syscall_names[1024] = {0};

/* Since this redfines "KILL_PROCESS" into a TRAP for the reporter hook,
 * we want to make sure it stands out in the build as it should not be
 * used in the final program.
 */
#undef KILL_PROCESS
#define KILL_PROCESS BPF_STMT(BPF_RET + BPF_K, SECCOMP_RET_TRAP)

const char *const msg_needed = "<<SANDBOXED>>::";

/* Since "sprintf" is technically not signal-safe, reimplement %d here. */
static void write_uint(char *buf, unsigned int val) {
  int width = 0;
  unsigned int tens;

  if (val == 0) {
    strcpy(buf, "0");
    return;
  }
  for (tens = val; tens; tens /= 10)
    ++width;
  buf[width] = '\0';
  for (tens = val; tens; tens /= 10)
    buf[--width] = '0' + (tens % 10);
}

static void wbSandbox_report(int nr, siginfo_t *info, void *void_context) {
  char buf[128];
  ucontext_t *ctx = (ucontext_t *)(void_context);
  unsigned int syscall;
  if (info->si_code != SYS_SECCOMP)
    return;
  if (!ctx)
    return;
  syscall = ctx->uc_mcontext.gregs[REG_SYSCALL];
  strcpy(buf, msg_needed);
  if (syscall < sizeof(syscall_names)) {
    strcat(buf, syscall_names[syscall]);
    strcat(buf, "(");
  }
  write_uint(buf + strlen(buf), syscall);
  if (syscall < sizeof(syscall_names))
    strcat(buf, ")");
  strcat(buf, "\n");
  size_t w = write(STDERR_FILENO, buf, strlen(buf));
  exit(1);
}

static int wbSandbox_reporter(void) {
  struct sigaction act;
  sigset_t mask;

#define wbSyscall_declare(n, s) syscall_names[n] = s;
#include <wbSyscallNames.inc.h>
#undef wbSyscall_declare

  memset(&act, 0, sizeof(act));
  sigemptyset(&mask);
  sigaddset(&mask, SIGSYS);

  act.sa_sigaction = &wbSandbox_report;
  act.sa_flags = SA_SIGINFO;
  if (sigaction(SIGSYS, &act, NULL) < 0) {
    perror("sigaction");
    return -1;
  }
  if (sigprocmask(SIG_UNBLOCK, &mask, NULL)) {
    perror("sigprocmask");
    return -1;
  }
  return 0;
}
#endif /* WB_USE_SANDBOX_DEBUG */

static int wbSandbox_filters(void) {
  struct sock_filter filter[] = {
      /* Validate architecture. */
      VALIDATE_ARCHITECTURE, /* Grab the system call number. */
      EXAMINE_SYSCALL,       /* List allowed syscalls. */
      ALLOW_SYSCALL(rt_sigreturn),
#ifdef __NR_sigreturn
      ALLOW_SYSCALL(sigreturn),
#endif
      ALLOW_SYSCALL(exit_group),
      ALLOW_SYSCALL(exit),
      ALLOW_SYSCALL(read),
      ALLOW_SYSCALL(write), /* Add more syscalls here. */
      ALLOW_SYSCALL(fstat),
      ALLOW_SYSCALL(lstat),
      ALLOW_SYSCALL(mmap),
      ALLOW_SYSCALL(rt_sigprocmask),
      ALLOW_SYSCALL(rt_sigaction),
      ALLOW_SYSCALL(nanosleep),
      ALLOW_SYSCALL(open),
      ALLOW_SYSCALL(close),
      ALLOW_SYSCALL(lseek),
      ALLOW_SYSCALL(munmap),
      ALLOW_SYSCALL(futex),
      ALLOW_SYSCALL(access),
      ALLOW_SYSCALL(mprotect),
      ALLOW_SYSCALL(sched_get_priority_max),
      ALLOW_SYSCALL(sched_get_priority_min),
      ALLOW_SYSCALL(geteuid),
      ALLOW_SYSCALL(ioctl),
      ALLOW_SYSCALL(uname),
      ALLOW_SYSCALL(sysinfo),
      ALLOW_SYSCALL(getrlimit),
      ALLOW_SYSCALL(brk),
      ALLOW_SYSCALL(pipe),
      ALLOW_SYSCALL(fcntl),
      ALLOW_SYSCALL(clone),
      ALLOW_SYSCALL(set_robust_list),
      ALLOW_SYSCALL(select),
      ALLOW_SYSCALL(mkdir),
      ALLOW_SYSCALL(stat),
      ALLOW_SYSCALL(readlink),
      KILL_PROCESS,
  };
  struct sock_fprog prog;

  prog.len = (unsigned short)(sizeof(filter) / sizeof(filter[0]));
  prog.filter = filter;

  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
    perror("prctl(NO_NEW_PRIVS)");
    goto failed;
  }
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)) {
    perror("prctl(SECCOMP)");
    goto failed;
  }
  return 0;

failed:
  if (errno == EINVAL)
    fprintf(stderr, "SECCOMP_FILTER is not available. :(\n");
  return 1;
}

int wbSandbox_new(void) {
#ifdef WB_USE_SANDBOX_DEBUG
  if (wbSandbox_reporter()) {
    return 1;
  }
#endif /* WB_USE_SANDBOX_DEBUG */
  if (wbSandbox_filters()) {
    return 1;
  }
  return 0;
}

#else  /* WB_USE_SANDBOX */
int wbSandbox_new(void) {
  // fprintf(stderr, "Not using sandbox mode.\n");
  return 0;
}
#endif /* WB_USE_SANDBOX */
