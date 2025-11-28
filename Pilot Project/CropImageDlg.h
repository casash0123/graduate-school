
// CropImageDlg.h: 헤더 파일
//

#pragma once


// CCropImageDlg 대화 상자
class CCropImageDlg : public CDialogEx
{
// 생성입니다.
public:
	CCropImageDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CROPIMAGE_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	CStatic m_Source;
	CStatic m_Target;
	CEdit   m_edit_X, m_edit_Y, m_edit_W, m_edit_H;

	CImage  m_imgSrc;
	CImage  m_imgCrop;

	CRect   m_rcImgAreaSrc;   // Static 전체
	CRect   m_rcImgDraw;      // 실제 이미지가 그려진 Rect (DrawSrcImage에서 세팅)

	// ★ 드래그 상태
	bool    m_bDragging = false;
	CPoint  m_ptStartImg;     // 드래그 시작점 (이미지 좌표)
	CPoint  m_ptCurImg;       // 드래그 현재점 (이미지 좌표)

	// ★ 이전에 그렸던 선택 박스 (STATIC_SOURCE 기준 좌표)
	CRect   m_rcPrevSelStatic;

	virtual BOOL PreTranslateMessage(MSG* pMsg);

	// 좌표 변환
	bool DlgPointToImagePoint(const CPoint& ptDlg, CPoint& ptImg);
	bool ImagePointToStaticPoint(const CPoint& ptImg, CPoint& ptStatic);

	// 박스 그리기 (STATIC_SOURCE 기준 좌표)
	void DrawSelectionRect(const CRect& rcStatic);

	void PasteImageFromClipboard();   // Ctrl+V
	void CopyCropToClipboard();
	void DrawSrcImage();              // 원본 표시
	void DrawCropImage();             // Crop 표시

	void RestoreSelectionFromEdit();
	void CropAndShow(int left, int top, int w, int h);

	afx_msg void OnBnClickedOk();
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
};
