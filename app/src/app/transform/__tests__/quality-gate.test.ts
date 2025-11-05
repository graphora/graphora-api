import { describe, expect, it } from 'vitest'

import { extractQualityGateInsight, isQualityGateWarning } from '../quality-gate'

describe('extractQualityGateInsight', () => {
  it('returns null when payload is undefined', () => {
    expect(extractQualityGateInsight(undefined)).toBeNull()
    expect(extractQualityGateInsight(null)).toBeNull()
  })

  it('extracts warning information from top-level fields', () => {
    const insight = extractQualityGateInsight({
      quality_gate_status: 'warn',
      quality_gate_reasons: ['Score below threshold'],
      quality_gate_message: 'Overall quality score dropped below the warning threshold.',
    })

    expect(insight).not.toBeNull()
    expect(insight?.status).toBe('warn')
    expect(insight?.reasons).toEqual(['Score below threshold'])
    expect(insight?.message).toContain('warning threshold')
  })

  it('extracts warning details from nested structures', () => {
    const payload = {
      failure_details: {
        gating_status: 'WARN',
        gating_reasons: 'Company property fill rate below 0.75',
      },
    }

    const insight = extractQualityGateInsight(payload)

    expect(insight).not.toBeNull()
    expect(insight?.status).toBe('warn')
    expect(insight?.reasons).toEqual(['Company property fill rate below 0.75'])
    expect(isQualityGateWarning(insight)).toBe(true)
  })

  it('ignores non-warning statuses', () => {
    const passInsight = extractQualityGateInsight({ quality_gate_status: 'pass' })
    expect(passInsight?.status).toBe('pass')
    expect(isQualityGateWarning(passInsight)).toBe(false)

    const failInsight = extractQualityGateInsight({ quality_gate_status: 'fail' })
    expect(failInsight?.status).toBe('fail')
    expect(isQualityGateWarning(failInsight)).toBe(false)
  })
})
