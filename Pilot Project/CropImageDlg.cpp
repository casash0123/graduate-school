
// CropImageDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "CropImage.h"
#include "CropImageDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CCropImageDlg 대화 상자



CCropImageDlg::CCropImageDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_CROPIMAGE_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCropImageDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);

	DDX_Control(pDX, IDC_STATIC_SOURCE, m_Source);
	DDX_Control(pDX, IDC_STATIC_TARGET, m_Target);

	DDX_Control(pDX, IDC_EDIT_X, m_edit_X);
	DDX_Control(pDX, IDC_EDIT_Y, m_edit_Y);
	DDX_Control(pDX, IDC_EDIT_W, m_edit_W);
	DDX_Control(pDX, IDC_EDIT_H, m_edit_H);
}

BEGIN_MESSAGE_MAP(CCropImageDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CCropImageDlg::OnBnClickedOk)
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()


// CCropImageDlg 메시지 처리기

BOOL CCropImageDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	CRect rc;
	m_Source.GetClientRect(&rc);
	m_rcImgAreaSrc = rc;

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CCropImageDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}


// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CCropImageDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CCropImageDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CCropImageDlg::OnBnClickedOk()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CDialogEx::OnOK();
}


BOOL CCropImageDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	if (pMsg->message == WM_KEYDOWN)
	{
		const bool ctrl = (::GetKeyState(VK_CONTROL) & 0x8000) != 0;
		if (ctrl && pMsg->wParam == 'V')
		{
			// Ctrl+V : 클립보드에서 이미지 붙여넣기
			PasteImageFromClipboard();
			return TRUE;
		}
		else if (ctrl && pMsg->wParam == 'C')
		{
			// Ctrl+C : Crop된 이미지 복사
			CopyCropToClipboard();
			return TRUE;
		}
	}

	return CDialogEx::PreTranslateMessage(pMsg);
}

bool CCropImageDlg::DlgPointToImagePoint(const CPoint& ptDlg, CPoint& ptImg)
{
	if (m_imgSrc.IsNull())
		return false;

	// 1) 다이얼로그 좌표 -> 화면 좌표
	CPoint ptScreen = ptDlg;
	ClientToScreen(&ptScreen);

	// 2) 화면 좌표 -> STATIC_SOURCE 클라이언트 좌표
	CRect rcSrcWnd;
	m_Source.GetWindowRect(&rcSrcWnd);

	CPoint ptStatic = ptScreen;
	ptStatic.Offset(-rcSrcWnd.left, -rcSrcWnd.top);

	// ★★★ "이미지 그려진 영역" 안인지 체크 (m_rcImgDraw 사용)
	if (!m_rcImgDraw.PtInRect(ptStatic))
		return false;

	// 4) 이미지 좌표로 매핑
	int imgW = m_imgSrc.GetWidth();
	int imgH = m_imgSrc.GetHeight();

	double sx = (double)imgW / m_rcImgDraw.Width();
	double sy = (double)imgH / m_rcImgDraw.Height();

	int relX = ptStatic.x - m_rcImgDraw.left;
	int relY = ptStatic.y - m_rcImgDraw.top;

	int x = (int)(relX * sx);
	int y = (int)(relY * sy);

	if (x < 0 || y < 0 || x >= imgW || y >= imgH)
		return false;

	ptImg.x = x;
	ptImg.y = y;
	return true;
}


bool CCropImageDlg::ImagePointToStaticPoint(const CPoint& ptImg, CPoint& ptStatic)
{
	if (m_imgSrc.IsNull())
		return false;

	int imgW = m_imgSrc.GetWidth();
	int imgH = m_imgSrc.GetHeight();
	if (imgW <= 0 || imgH <= 0)
		return false;

	double sx = (double)m_rcImgDraw.Width() / imgW;
	double sy = (double)m_rcImgDraw.Height() / imgH;

	int x = m_rcImgDraw.left + (int)(ptImg.x * sx);
	int y = m_rcImgDraw.top + (int)(ptImg.y * sy);

	ptStatic.x = x;
	ptStatic.y = y;
	return true;
}

void CCropImageDlg::DrawSelectionRect(const CRect& rcStatic)
{
	if (rcStatic.IsRectEmpty())
		return;

	CClientDC dc(&m_Source);
	CRect rc = rcStatic;
	dc.DrawFocusRect(&rc);   // XOR 박스
}

void CCropImageDlg::PasteImageFromClipboard()
{
	if (!::OpenClipboard(m_hWnd))
		return;

	HBITMAP hBmp = (HBITMAP)::GetClipboardData(CF_BITMAP);
	if (hBmp == nullptr)
	{
		::CloseClipboard();
		return;
	}

	// 기존 이미지 해제
	if (!m_imgSrc.IsNull())
		m_imgSrc.Destroy();

	// CImage로 Attach
	m_imgSrc.Attach(hBmp);

	// 표시
	DrawSrcImage();

	::CloseClipboard();

	// ★ 여기! Ctrl+V 후, EDIT 값 기준으로 박스/크롭 복원
	RestoreSelectionFromEdit();

	CopyCropToClipboard();
}

void CCropImageDlg::CopyCropToClipboard()
{
	if (m_imgCrop.IsNull())
		return;

	int w = m_imgCrop.GetWidth();
	int h = m_imgCrop.GetHeight();
	if (w <= 0 || h <= 0)
		return;

	// 화면 DC 가져오기
	HDC hScreenDC = ::GetDC(nullptr);
	if (hScreenDC == nullptr)
		return;

	CDC dcScreen;
	dcScreen.Attach(hScreenDC);

	CDC dcSrc, dcDst;
	dcSrc.CreateCompatibleDC(&dcScreen);
	dcDst.CreateCompatibleDC(&dcScreen);

	HBITMAP hBmpSrc = (HBITMAP)m_imgCrop;
	HBITMAP hOldSrc = (HBITMAP)dcSrc.SelectObject(hBmpSrc);

	// ★ 여기 수정: &dcScreen 말고 dcScreen 또는 GetSafeHdc()
	HBITMAP hBmpCopy = ::CreateCompatibleBitmap(dcScreen, w, h);
	// 또는: HBITMAP hBmpCopy = ::CreateCompatibleBitmap(dcScreen.GetSafeHdc(), w, h);

	if (hBmpCopy == nullptr)
	{
		dcSrc.SelectObject(hOldSrc);
		dcScreen.Detach();
		::ReleaseDC(nullptr, hScreenDC);
		return;
	}

	HBITMAP hOldDst = (HBITMAP)dcDst.SelectObject(hBmpCopy);

	// 내용 복사
	dcDst.BitBlt(0, 0, w, h, &dcSrc, 0, 0, SRCCOPY);

	// 원복
	dcSrc.SelectObject(hOldSrc);
	dcDst.SelectObject(hOldDst);

	dcScreen.Detach();
	::ReleaseDC(nullptr, hScreenDC);

	// 클립보드로 넘기기
	if (!::OpenClipboard(m_hWnd))
	{
		::DeleteObject(hBmpCopy);
		return;
	}

	::EmptyClipboard();
	::SetClipboardData(CF_BITMAP, hBmpCopy); // 소유권 이동
	::CloseClipboard();
}




void CCropImageDlg::DrawSrcImage()
{
	if (m_imgSrc.IsNull())
		return;

	CClientDC dc(&m_Source);

	// Static 크기
	CRect rc;
	m_Source.GetClientRect(&rc);
	int dstW = rc.Width();
	int dstH = rc.Height();

	// 원본 크기
	int srcW = m_imgSrc.GetWidth();
	int srcH = m_imgSrc.GetHeight();

	// ===== 비율 계산 =====
	double ratioW = (double)dstW / srcW;
	double ratioH = (double)dstH / srcH;
	double ratio = min(ratioW, ratioH);  // 비율 유지!

	int drawW = (int)(srcW * ratio);
	int drawH = (int)(srcH * ratio);

	int offsetX = (dstW - drawW) / 2;
	int offsetY = (dstH - drawH) / 2;

	CRect drawRect(offsetX, offsetY, offsetX + drawW, offsetY + drawH);

	// ★★★ 여기 추가: 실제 이미지 그려진 영역 저장
	m_rcImgDraw = drawRect;

	// ===== 비율 고정 Draw =====
	m_imgSrc.Draw(dc, drawRect);
}

void CCropImageDlg::DrawCropImage()
{
	if (m_imgCrop.IsNull())
		return;

	CClientDC dcCrop(&m_Target);

	// TARGET Static 크기
	CRect rcClient;
	m_Target.GetClientRect(&rcClient);
	int dstW = rcClient.Width();
	int dstH = rcClient.Height();

	// 크롭 이미지 원본 크기
	int srcW = m_imgCrop.GetWidth();
	int srcH = m_imgCrop.GetHeight();
	if (srcW <= 0 || srcH <= 0)
		return;

	// 비율 유지해서 맞출 스케일
	double ratioW = (double)dstW / srcW;
	double ratioH = (double)dstH / srcH;
	double ratio = min(ratioW, ratioH);

	int drawW = (int)(srcW * ratio);
	int drawH = (int)(srcH * ratio);

	// 중앙 정렬 (좌우/상하 여백 자동)
	int offsetX = (dstW - drawW) / 2;
	int offsetY = (dstH - drawH) / 2;

	CRect drawRect(
		rcClient.left + offsetX,
		rcClient.top + offsetY,
		rcClient.left + offsetX + drawW,
		rcClient.top + offsetY + drawH
	);

	// ★ 여기서 배경 먼저 싹 지우고
	dcCrop.FillSolidRect(&rcClient, ::GetSysColor(COLOR_3DFACE));
	// 또는 RGB(0,0,0) 같은 고정 색 써도 됨
	// dcCrop.FillSolidRect(&rcClient, RGB(0, 0, 0));

	// 비율 고정 + 중앙 배치로 그리기
	m_imgCrop.Draw(dcCrop, drawRect);
}

void CCropImageDlg::RestoreSelectionFromEdit()
{
	if (m_imgSrc.IsNull())
		return;

	// 1) Edit에서 X, Y, W, H 읽기
	CString sx, sy, sw, sh;
	m_edit_X.GetWindowText(sx);
	m_edit_Y.GetWindowText(sy);
	m_edit_W.GetWindowText(sw);
	m_edit_H.GetWindowText(sh);

	int left = _ttoi(sx);
	int top = _ttoi(sy);
	int w = _ttoi(sw);
	int h = _ttoi(sh);

	if (w <= 0 || h <= 0)
		return;

	int imgW = m_imgSrc.GetWidth();
	int imgH = m_imgSrc.GetHeight();
	if (imgW <= 0 || imgH <= 0)
		return;

	// 2) 이미지 범위로 클램핑
	if (left < 0) left = 0;
	if (top < 0) top = 0;
	if (left + w > imgW)  w = imgW - left;
	if (top + h > imgH)  h = imgH - top;
	if (w <= 0 || h <= 0)
		return;

	// 3) 이미지 좌표 → STATIC_SOURCE 좌표로 변환
	CPoint ptImgLT(left, top);
	CPoint ptImgRB(left + w, top + h);
	CPoint ptStaticLT, ptStaticRB;

	if (!ImagePointToStaticPoint(ptImgLT, ptStaticLT))
		return;
	if (!ImagePointToStaticPoint(ptImgRB, ptStaticRB))
		return;

	CRect rcStatic(ptStaticLT, ptStaticRB);
	rcStatic.NormalizeRect();

	// 4) 이전 박스 값 갱신 후 다시 그리기
	m_rcPrevSelStatic = rcStatic;
	DrawSelectionRect(m_rcPrevSelStatic);

	// 5) TARGET 쪽 크롭 이미지도 다시 갱신
	CropAndShow(left, top, w, h);
}


// CropImageDlg.cpp

void CCropImageDlg::CropAndShow(int left, int top, int w, int h)
{
	if (m_imgSrc.IsNull())
		return;

	int imgW = m_imgSrc.GetWidth();
	int imgH = m_imgSrc.GetHeight();
	if (imgW <= 0 || imgH <= 0)
		return;

	// 1) 이미지 범위 안으로 클램핑
	if (left < 0) left = 0;
	if (top < 0) top = 0;
	if (left + w > imgW)  w = imgW - left;
	if (top + h > imgH)  h = imgH - top;
	if (w <= 0 || h <= 0)
		return;

	// 2) 기존 크롭 이미지 해제
	if (!m_imgCrop.IsNull())
		m_imgCrop.Destroy();

	// 3) 새 CImage 생성 (w x h 크기로)
	m_imgCrop.Create(w, h, m_imgSrc.GetBPP());

	// 4) BitBlt로 잘라오기
	CDC dcSrc, dcDst;
	dcSrc.CreateCompatibleDC(nullptr);
	dcDst.CreateCompatibleDC(nullptr);

	HBITMAP hBmpSrc = (HBITMAP)m_imgSrc;
	HBITMAP hBmpDst = (HBITMAP)m_imgCrop;

	HBITMAP hOldSrc = (HBITMAP)dcSrc.SelectObject(hBmpSrc);
	HBITMAP hOldDst = (HBITMAP)dcDst.SelectObject(hBmpDst);

	dcDst.BitBlt(0, 0, w, h, &dcSrc, left, top, SRCCOPY);

	dcSrc.SelectObject(hOldSrc);
	dcDst.SelectObject(hOldDst);

	// 5) TARGET Static에 그리기
	DrawCropImage();
}



void CCropImageDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// 먼저 이 좌표가 STATIC_SOURCE 위인지 체크
	CPoint ptScreen = point;
	ClientToScreen(&ptScreen);

	CRect rcSourceWnd;
	m_Source.GetWindowRect(&rcSourceWnd);

	if (!rcSourceWnd.PtInRect(ptScreen))
	{
		CDialogEx::OnLButtonDown(nFlags, point);
		return;
	}

	// ★★★ 이전 박스 있으면 지우고 초기화
	if (!m_rcPrevSelStatic.IsRectEmpty())
	{
		DrawSelectionRect(m_rcPrevSelStatic);   // XOR로 한 번 더 그려서 제거
		m_rcPrevSelStatic.SetRectEmpty();
	}

	CPoint ptImg;
	if (DlgPointToImagePoint(point, ptImg))
	{
		m_bDragging = true;
		m_ptStartImg = ptImg;
		m_ptCurImg = ptImg;

		SetCapture();
	}

	CDialogEx::OnLButtonDown(nFlags, point);
}


void CCropImageDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if (!m_bDragging)
		return;

	m_bDragging = false;
	ReleaseCapture();

	// ★ 박스 지우지 않으려면 이 부분은 주석/삭제
	// if (!m_rcPrevSelStatic.IsRectEmpty())
	// {
	//     DrawSelectionRect(m_rcPrevSelStatic);
	//     m_rcPrevSelStatic.SetRectEmpty();
	// }

	CPoint ptEndImg;
	if (!DlgPointToImagePoint(point, ptEndImg))
		return;

	int x1 = m_ptStartImg.x;
	int y1 = m_ptStartImg.y;
	int x2 = ptEndImg.x;
	int y2 = ptEndImg.y;

	int left = min(x1, x2);
	int top = min(y1, y2);
	int right = max(x1, x2);
	int bottom = max(y1, y2);

	int w = right - left;
	int h = bottom - top;
	if (w <= 0 || h <= 0)
		return;

	// EDIT에 기록
	CString sx, sy, sw, sh;
	sx.Format(_T("%d"), left);
	sy.Format(_T("%d"), top);
	sw.Format(_T("%d"), w);
	sh.Format(_T("%d"), h);

	m_edit_X.SetWindowText(sx);
	m_edit_Y.SetWindowText(sy);
	m_edit_W.SetWindowText(sw);
	m_edit_H.SetWindowText(sh);

	// ★★★ 여기! UP 할 때 선택 영역 크롭해서 STATIC_TARGET에 표시
	CropAndShow(left, top, w, h);

	CDialogEx::OnLButtonUp(nFlags, point);
}
void CCropImageDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	CDialogEx::OnMouseMove(nFlags, point);

	if (!m_bDragging)
		return;

	// 현재 위치를 이미지 좌표로
	CPoint ptImg;
	if (!DlgPointToImagePoint(point, ptImg))
		return;

	m_ptCurImg = ptImg;

	// 이미지 좌표 기준 Rect
	int x1 = m_ptStartImg.x;
	int y1 = m_ptStartImg.y;
	int x2 = m_ptCurImg.x;
	int y2 = m_ptCurImg.y;

	int leftImg = min(x1, x2);
	int topImg = min(y1, y2);
	int rightImg = max(x1, x2);
	int bottomImg = max(y1, y2);

	CPoint ptStaticLT, ptStaticRB;
	if (!ImagePointToStaticPoint(CPoint(leftImg, topImg), ptStaticLT))
		return;
	if (!ImagePointToStaticPoint(CPoint(rightImg, bottomImg), ptStaticRB))
		return;

	CRect rcStatic(ptStaticLT, ptStaticRB);
	rcStatic.NormalizeRect();

	// 1) 이전 박스 지우기
	if (!m_rcPrevSelStatic.IsRectEmpty())
		DrawSelectionRect(m_rcPrevSelStatic);

	// 2) 새 박스 그리기
	DrawSelectionRect(rcStatic);

	// 3) 현재 박스 저장
	m_rcPrevSelStatic = rcStatic;
}
